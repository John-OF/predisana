import json
import os
import sys
import csv
import io
import re
import math
import hmac
import time
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Dict, Any, List, Optional

import glob

# En Windows la consola usa cp1252 y revienta al imprimir emojis (🎲, ⚠️) en los
# logs. Forzamos UTF-8 en stdout/stderr para que esos print() no tumben requests.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

import shap

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from joblib import load
from sklearn.linear_model import LogisticRegression

from sqlalchemy import (
    create_engine, inspect, text, func,
    Column, Integer, String, Float, Text as SAText, DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import coherence
import modos as M
import monotonic_logreg  # noqa: F401  las LogReg servidas la referencian al cargarse
import synthetic_quality as sq
from risk_banding import band_cutoffs, band_of

EXPLAINERS: Dict[str, Any] = {}

app = Flask(__name__)

# ==========================================
# CORS Y RATE LIMITING (AUD-4)
# ==========================================
def _origenes_desde_env(var: str, defecto: List[str]) -> List[str]:
    """Lee una lista de origenes separada por comas de una env var."""
    crudo = os.environ.get(var, "")
    lista = [o.strip() for o in crudo.split(",") if o.strip()]
    return lista or defecto


# Origenes de desarrollo: el dev server de Vite (5173) y el preview del build (4173).
DEV_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
]
# Endpoints publicos: abiertos si no se configura nada (comodo en dev y para probar
# el API con curl). En el deploy (#7) se setea CORS_ORIGINS al dominio del front.
CORS_ORIGINS = _origenes_desde_env("CORS_ORIGINS", ["*"])
# El admin NO hereda ese "*". Con "*" el navegador deja que CUALQUIER pagina lea la
# respuesta de /admin/*, asi que una pagina hostil podria probar tokens desde el
# navegador del dev y leer el resultado. Restringido, el navegador bloquea la lectura.
# (Ojo: CORS no protege de un atacante que llame al API directamente sin navegador;
# de eso se encarga el rate limiting de abajo.)
ADMIN_CORS_ORIGINS = _origenes_desde_env("ADMIN_CORS_ORIGINS", DEV_ORIGINS)
# flask-cors resuelve el recurso mas especifico primero, asi que /admin/* gana sobre /*.
CORS(app, resources={
    r"/admin/*": {"origins": ADMIN_CORS_ORIGINS},
    r"/*": {"origins": CORS_ORIGINS},
})

# Detras de un proxy (Vercel/Render/Fly) request.remote_addr es la IP del proxy: sin
# esto TODO el trafico compartiria la misma cubeta y el limite seria inservible. Es
# opt-in porque confiar en X-Forwarded-For SIN un proxy delante permite falsear la IP
# y saltarse el limite a voluntad.
if os.environ.get("TRUST_PROXY_HEADERS", "").lower() in ("1", "true", "yes"):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Limites configurables por env var (el deploy puede aflojarlos o apretarlos sin tocar
# codigo). RATE_LIMIT_ADMIN_VERIFY es el importante: los intentos FALLIDOS de
# ADMIN_TOKEN por IP, sumando los cuatro endpoints del panel (_intentos_token_admin).
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "300 per minute")
RATE_LIMIT_PREDICT = os.environ.get("RATE_LIMIT_PREDICT", "60 per minute")
RATE_LIMIT_ADMIN = os.environ.get("RATE_LIMIT_ADMIN", "60 per minute")
RATE_LIMIT_ADMIN_VERIFY = os.environ.get("RATE_LIMIT_ADMIN_VERIFY", "10 per minute")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT_DEFAULT],
    storage_uri=os.environ.get("RATE_LIMIT_STORAGE", "memory://"),
    strategy="fixed-window",
    headers_enabled=True,
)


@app.errorhandler(429)
def _demasiadas_peticiones(e):
    return jsonify({
        "error": "demasiadas peticiones, intenta de nuevo en un momento",
        "limite": str(getattr(e, "description", "")),
    }), 429

# AUD-9: los payloads legitimos son de unos cientos de bytes (un puñado de
# features numericas). Sin tope, cualquiera puede mandar un cuerpo gigante y
# obligar al servidor a bufferearlo entero.
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # 256 KB


@app.errorhandler(413)
def _cuerpo_demasiado_grande(_e):
    return jsonify({"error": "cuerpo demasiado grande"}), 413

# Rutas ancladas a la carpeta de este archivo, no al directorio de trabajo: arrancada
# desde otra carpeta (la raiz del repo, o un deploy sin `--chdir`) la app levantaba
# sin modelos ni datos, todo /predict daba 500 y /health decia "ok".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS = os.path.join(BASE_DIR, "models")
DATA_CURATED = os.path.join(BASE_DIR, "data_curated")
DATA_PROCESSED = os.path.join(BASE_DIR, "data_processed")
DB_PATH = os.path.join(BASE_DIR, "medical_history.db")
# Capa de datos agnóstica al motor (A3): SQLite en dev, Postgres en prod (#7),
# mismo código. Se controla con la env var DATABASE_URL (vacia = la SQLite de aqui).
DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///" + DB_PATH.replace(os.sep, "/")
# Token del panel admin dev-only (A3). Si no está seteado, el admin queda
# deshabilitado (los endpoints /admin/* responden 503). NO es auth de usuario:
# los usuarios nunca se loguean, las simulaciones son anónimas.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

# ==========================================
# MODELOS: cada enfermedad, dos modos (v2, revision 2026-10)
# ==========================================
# Cada enfermedad tiene dos modelos: el SIMPLIFICADO (lo que cualquiera sabe de si
# mismo: edad, sexo, peso y talla, tabaco y diagnosticos conocidos) y el COMPLETO
# (ademas lo que mide el personal sanitario). Los dos estiman la enfermedad TOTAL
# (diagnosticada o detectada por analitica o medicion) y salen de NHANES 2017-2023
# (prepare_nhanes.py + train_models.py). Que variables usa cada uno vive en modos.py.
ENFERMEDADES = list(M.ENFERMEDADES)
MODOS = list(M.MODOS)
MODO_POR_DEFECTO = "simplificado"

MODELS: Dict[str, Any] = {}             # clave (enfermedad_modo) -> pipeline
FEATURES: Dict[str, List[str]] = {}
MODEL_NAMES: Dict[str, str] = {}        # clave -> modelo ganador del bake-off
CALIBRATORS: Dict[str, Any] = {}        # clave -> isotonica centrada
# Variables que el modelo servido no usa: las restricciones de monotonia las dejaron
# en cero porque los datos no dan senal en el sentido clinico (train_models.py las
# registra en las metricas). Se aceptan, pero hay que decir que no cuentan.
SIN_EFECTO: Dict[str, List[str]] = {}
MEDIANAS: Dict[str, Dict[str, float]] = {}  # clave -> mediana del train de las sin efecto


def _clave(disease: str, modo: str) -> str:
    return f"{disease}_{modo}"


def _partes(clave: str):
    disease, modo = clave.rsplit("_", 1)
    return disease, modo


CLAVES = [_clave(d, m) for d in ENFERMEDADES for m in MODOS]


def _paths(clave: str) -> Dict[str, str]:
    return {k: os.path.join(BASE_MODELS, f"{clave}_{k}.{ext}") for k, ext in (
        ("pipeline", "pkl"), ("calibrator", "pkl"), ("features", "json"), ("metrics", "json"))}


# Datos que se interpretan con las guias (ADA, ACC/AHA, KDIGO) en la capa clinica.
# Cuando definen la enfermedad (HbA1c y glucosa en diabetes, presion en hipertension,
# filtrado y albumina en la renal) no entran al modelo: con ellos el modelo solo
# reaprenderia el umbral diagnostico. Se pueden aportar en los dos modos, y en el
# completo el formulario los pide.
ENTRADAS_CLINICAS = ("blood_glucose_level", "hba1c_level", "ap_hi", "ap_lo", "egfr",
                     "albumin_creatinine_ratio")
# Las que el formulario ofrece como opcionales en cualquier enfermedad. La funcion
# renal, solo donde la define (o donde es variable del modelo): en el simplificado de
# diabetes un filtrado glomerular seria un campo mas que casi nadie conoce.
CLINICAS_DE_SIEMPRE = ("blood_glucose_level", "hba1c_level", "ap_hi", "ap_lo")


def _clinicas_del_formulario(disease: str, feats: List[str]) -> List[str]:
    return [c for c in ENTRADAS_CLINICAS if c not in feats
            and (c in CLINICAS_DE_SIEMPRE or c in M.DEFINITORIAS[disease])]

# Limites FISICOS de cada dato de entrada: lo que se acepta, no lo que el modelo vio
# (eso es SUPPORT, AUD-16, y solo avisa). Fuera de aqui no hay paciente posible
# (edad -30, IMC 900) y antes se aceptaba con un 200, probabilidad 1.0 y fila en la
# BD. Cubren todo el rango de los datos reales y sinteticos (hay un test que lo
# exige), y /config los sirve como `ranges` para que el formulario valide con los
# MISMOS numeros.
_BINARIAS = [0, 1]
INPUT_LIMITS: Dict[str, List[float]] = {
    "age": [18, 100],
    "weight": [25, 300],
    "height": [100, 230],
    "bmi": [10, 95],
    "waist_circumference": [40, 200],
    "ap_hi": [60, 260],
    "ap_lo": [30, 160],
    "total_cholesterol": [50, 600],
    "hdl_cholesterol": [5, 250],
    "hba1c_level": [2.5, 20],
    "blood_glucose_level": [30, 700],
    "egfr": [1, 200],
    "albumin_creatinine_ratio": [0, 20000],
    "alt": [1, 2000],
    **{b: _BINARIAS for b in ("hypertension", "heart_disease", "diabetes", "high_cholesterol")},
}
_PREFIJOS_ONE_HOT = ("gender_", "smoking_history_")
_NOMBRE_GRUPO = {"gender": "sexo", "smoking_history": "tabaquismo"}


def _input_limits(feature: str) -> Optional[List[float]]:
    if feature.startswith(_PREFIJOS_ONE_HOT):
        return _BINARIAS
    return INPUT_LIMITS.get(feature)


# NHANES registra la edad con TOPE: todo el que pasa de 80 figura como 80 (RIDAGEYR).
# El modelo si vio gente de 85, pero codificada como 80, asi que por encima del tope
# el aviso no puede decir que "nunca vio un caso asi".
TOPCODED: Dict[str, Dict[str, float]] = {d: {"age": 80} for d in ENFERMEDADES}

# ==========================================
# 1. BASE DE DATOS (SQLAlchemy, agnóstica al motor — A3)
# ==========================================
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)
Base = declarative_base()


class Prediction(Base):
    """Log anónimo de cada simulación. Sin PII: solo inputs de salud + salida."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    disease = Column(String(50), index=True)
    input_data = Column(SAText)                  # JSON con las features enviadas
    prediction = Column(Integer)                 # clase 0/1
    probability = Column(Float)                  # salida limpia del modelo
    model_name = Column(String(50))              # A5: modelo ganador servido
    mode = Column(String(20))                    # v2: simplificado | completo
    clinical_note = Column(SAText)               # A4: nota clínica ADA/ACC-AHA
    top_features = Column(SAText)                # JSON con el SHAP top
    session_id = Column(String(64), index=True)  # UUID anónimo (agrupa sin identificar)
    timestamp = Column(DateTime, server_default=func.now())


def _migrate_add_columns():
    """Añade columnas nuevas a una tabla `predictions` preexistente (esquema viejo
    de 5 campos). create_all NO altera tablas existentes; sqlite y Postgres ambos
    soportan ALTER TABLE ADD COLUMN."""
    insp = inspect(engine)
    if "predictions" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("predictions")}
    wanted = {
        "model_name": "VARCHAR(50)",
        "clinical_note": "TEXT",
        "top_features": "TEXT",
        "session_id": "VARCHAR(64)",
        "mode": "VARCHAR(20)",
    }
    with engine.begin() as conn:
        for col, ddl in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col} {ddl}"))


def init_db():
    """Crea la tabla si no existe y migra columnas nuevas si venía del esquema viejo."""
    Base.metadata.create_all(engine)
    _migrate_add_columns()


def log_prediction_to_db(disease, input_data, prediction, probability,
                         model_name=None, clinical_note=None,
                         top_features=None, session_id=None, mode=None):
    """Guarda el historial de uso (anónimo). Nunca debe tumbar un request."""
    try:
        with SessionLocal() as s:
            s.add(Prediction(
                disease=disease,
                input_data=json.dumps(input_data, ensure_ascii=False),
                prediction=int(prediction),
                probability=float(probability),
                model_name=model_name,
                clinical_note=clinical_note,
                top_features=(json.dumps(top_features, ensure_ascii=False)
                              if top_features is not None else None),
                session_id=session_id,
                mode=mode,
            ))
            s.commit()
    except Exception as e:
        print(f"[WARN] Error guardando en BD: {e}")


class InvalidPayload(ValueError):
    """Valor de entrada no numerico o no finito. Se traduce a HTTP 400 (antes
    reventaba en `np.array(dtype=float)` como un 500 sin control)."""


def _json_safe(val):
    """Tipos numpy -> nativos y NaN/inf -> None. Flask serializa NaN como el
    literal `NaN`, que NO es JSON valido y rompe el JSON.parse del navegador
    (los datos reales de NHANES traen labs ausentes)."""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, float)):
        f = float(val)
        return round(f, 2) if math.isfinite(f) else None
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _to_float(feature: str, val):
    """Convierte un valor del payload a float o lanza InvalidPayload (-> 400)."""
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (list, dict, tuple, set)):
        raise InvalidPayload(f"'{feature}' debe ser un numero, no una lista u objeto")
    try:
        f = float(val)
    except (TypeError, ValueError):
        raise InvalidPayload(f"'{feature}' debe ser numerico (recibido: {val!r})")
    if not math.isfinite(f):
        raise InvalidPayload(f"'{feature}' debe ser un numero finito")
    return f


def _to_valid_input(feature: str, val) -> float:
    """_to_float + limites fisicos (INPUT_LIMITS). Solo para lo que ENTRA al modelo o
    a la capa clinica; el log y los avisos reutilizan valores ya validados."""
    f = _to_float(feature, val)
    limites = _input_limits(feature)
    if limites is not None and not (limites[0] <= f <= limites[1]):
        raise InvalidPayload(
            f"'{feature}' fuera de rango: se acepta de {limites[0]:g} a {limites[1]:g} "
            f"(recibido: {f:g})")
    return f


def _loggable_payload(clave: str, datos: Dict[str, Any]) -> Dict[str, float]:
    """Recorta el payload a lo que el modelo (o la capa clinica) usa de verdad: sus
    features, el peso y la talla de los que sale el IMC, y los datos clinicos. Antes
    se guardaba el JSON entero tal cual: claves arbitrarias del cliente engordando la
    BD sin aportar nada, y texto libre en una tabla que se exporta."""
    disease, modo = _partes(clave)
    permitidas = set(FEATURES.get(clave, [])) | set(ENTRADAS_CLINICAS)
    permitidas |= set(M.DERIVADAS.get((modo, "bmi"), ()))
    limpio = {}
    for k, v in datos.items():
        if k not in permitidas:
            continue
        try:
            limpio[k] = _to_float(k, v)
        except InvalidPayload:
            continue  # valor raro en una clave opcional: se descarta del log
    return limpio


_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _clean_session_id(raw):
    """El header X-Session-Id lo controla el cliente: se acepta solo un id corto y
    sano. Evita romper VARCHAR(64) en Postgres (donde el INSERT fallaria en
    silencio) y de paso cierra la via de inyeccion de formulas en el CSV."""
    if not raw:
        return None
    sid = _SESSION_ID_RE.sub("", str(raw))[:64]
    return sid or None


def _csv_safe(val):
    """Antepone una comilla a las celdas que empiezan por = + - @ (o control):
    Excel/Sheets las interpretarian como formula (inyeccion via campos de texto
    controlados por el cliente)."""
    txt = "" if val is None else str(val)
    if txt[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + txt
    return txt


def _safe_json_loads(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


# ==========================================
# 1b. AUTH DEL PANEL ADMIN (dev-only — A3)
# ==========================================
def require_admin(fn):
    """Protege los endpoints /admin/* con un token en header X-Admin-Token.
    NO es auth de usuario: solo el dev. Si ADMIN_TOKEN no está configurado, el
    panel queda deshabilitado (503)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not ADMIN_TOKEN:
            return jsonify({"error": "admin deshabilitado: configura ADMIN_TOKEN"}), 503
        # AUD-4: CORS solo *oculta* la respuesta al navegador; la peticion igual se
        # ejecuta. Rechazando aqui los Origin no permitidos, una pagina hostil no
        # llega ni a probar tokens desde el navegador del dev. Sin cabecera Origin
        # (curl, el propio deploy) se sigue permitiendo: ahi manda el token.
        origen = request.headers.get("Origin")
        if origen and "*" not in ADMIN_CORS_ORIGINS and origen not in ADMIN_CORS_ORIGINS:
            return jsonify({"error": "origen no permitido (revisa ADMIN_CORS_ORIGINS)"}), 403
        # compare_digest: comparacion en tiempo constante (sin canal lateral por
        # tiempo, a diferencia de `!=`, que corta en el primer caracter distinto).
        enviado = (request.headers.get("X-Admin-Token") or "").encode("utf-8")
        if not hmac.compare_digest(enviado, ADMIN_TOKEN.encode("utf-8")):
            return jsonify({"error": "no autorizado"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ==========================================
# DATOS DE ENTRENAMIENTO DE CADA MODO
# ==========================================
@lru_cache(maxsize=16)
def _train_modo(clave: str) -> pd.DataFrame:
    """Train real de un modelo con los nombres de SUS features (el simplificado lee el
    IMC autodeclarado como `bmi`), solo filas completas, mas el peso y la talla del
    simplificado y el objetivo. Alimenta el fondo de SHAP, la cobertura de datos y
    las medianas. OJO: el DataFrame es compartido, quien lo use no debe mutarlo."""
    disease, modo = _partes(clave)
    df = pd.read_csv(os.path.join(DATA_CURATED, disease, f"{disease}_train.csv"), low_memory=False)
    feats = M.features(disease, modo)
    out = pd.DataFrame({f: df[M.columna(modo, f)] for f in feats})
    for extra in M.DERIVADAS.get((modo, "bmi"), ()):
        out[extra] = df[extra]
    out["target"] = df["target"]
    return out.dropna(subset=feats).reset_index(drop=True)


# >>> SHAP START
def _build_shap_explainer(clave: str):
    """LinearExplainer para la LogReg y TreeExplainer para LightGBM. Los dos operan
    sobre lo que ve el clasificador: la salida de los pasos previos del pipeline
    (escala log de la albumina, escalado)."""
    pipe = MODELS[clave]
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        fondo = _train_modo(clave)[FEATURES[clave]]
        fondo = fondo.sample(min(200, len(fondo)), random_state=42).values.astype(float)
        EXPLAINERS[clave] = shap.LinearExplainer(
            clf, pipe[:-1].transform(fondo), feature_perturbation="interventional")
    else:
        # tree_path_dependent, sin fondo: exacto y sin los falsos fallos del
        # "additivity check" que da LightGBM con fondo.
        EXPLAINERS[clave] = shap.TreeExplainer(clf)


def _shap_fila(clave: str, X: np.ndarray) -> Optional[np.ndarray]:
    """Valores SHAP de una fila para la clase positiva, normalizando las formas que
    devuelven los explainers (lista por clase, 2D o 3D)."""
    explainer = EXPLAINERS.get(clave)
    if explainer is None:
        return None
    Xt = MODELS[clave][:-1].transform(X)
    if isinstance(explainer, shap.TreeExplainer):
        # check_additivity desactivado: solo se ordena por |SHAP|.
        sv = explainer.shap_values(Xt, check_additivity=False)
    else:
        sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv.reshape(-1)
# >>> SHAP END


# ==========================================
# CARGA DE MODELOS
# ==========================================
def _modelos_faltantes() -> List[str]:
    """Modelos que la app tiene que servir y no estan cargados: sin ellos, /predict
    de ese modo no puede responder."""
    return sorted(c for c in CLAVES if c not in MODELS or c not in FEATURES)


def _load_all():
    for clave in CLAVES:
        p = _paths(clave)
        if os.path.exists(p["pipeline"]):
            MODELS[clave] = load(p["pipeline"])
        if os.path.exists(p["features"]):
            with open(p["features"], "r", encoding="utf-8") as f:
                FEATURES[clave] = json.load(f)
        if os.path.exists(p["metrics"]):
            try:
                with open(p["metrics"], "r", encoding="utf-8") as f:
                    metricas = json.load(f)
                MODEL_NAMES[clave] = metricas.get("best_model")
                SIN_EFECTO[clave] = list(metricas.get("sin_efecto") or [])
            except Exception:
                pass
        if os.path.exists(p["calibrator"]):
            try:
                CALIBRATORS[clave] = load(p["calibrator"])
            except Exception as e:
                print(f"⚠️ Calibrador no cargado para {clave}: {e}")
    faltan = _modelos_faltantes()
    if faltan:
        print(f"⚠️ Modelos sin cargar: {', '.join(faltan)} (buscados en {BASE_MODELS}); "
              f"/health respondera 503.")
    for clave in MODELS:
        try:
            _build_shap_explainer(clave)
        except Exception as e:
            print(f"⚠️ SHAP no disponible para {clave}: {e}")
        try:
            SUPPORT[clave] = _build_support_stats(clave)
            tr = _train_modo(clave)
            MEDIANAS[clave] = {f: float(tr[f].median()) for f in SIN_EFECTO.get(clave, [])}
        except Exception as e:
            print(f"⚠️ Cobertura de datos no disponible para {clave}: {e}")
    # Prevalencia real de cada enfermedad, de la que salen las bandas de riesgo (las
    # comparten los dos modos). Sin ella quedan los tercios fijos.
    for disease in ENFERMEDADES:
        try:
            prevalencia = _build_prevalence(disease)
            if prevalencia is not None:
                PREVALENCE[disease] = prevalencia
        except Exception as e:
            print(f"⚠️ Prevalencia no disponible para {disease}: {e}")


def _safe_get(payload: Dict[str, Any], key: str):
    if key in payload: return payload[key]
    if " " in key:
        alt = key.replace(" ", "_")
        if alt in payload: return payload[alt]
    low = {k.lower(): v for k, v in payload.items()}
    if key.lower() in low: return low[key.lower()]
    if " " in key and key.replace(" ", "_").lower() in low:
        return low[key.replace(" ", "_").lower()]
    return None


def _vacio(val) -> bool:
    """Ausente = no viene o viene vacio (un input borrado en el form manda "")."""
    return val is None or (isinstance(val, str) and not val.strip())


# ==========================================
# CAPA DE INTERPRETACIÓN CLÍNICA (A4)
# ==========================================
# Umbrales diagnósticos de referencia, expuestos como una capa SEPARADA y
# etiquetada que se muestra JUNTO a la probabilidad del modelo, NO encima de
# ella. La probabilidad reportada es la salida limpia del modelo de ML; estos
# indicadores no la modifican (a diferencia del antiguo `max()` con números
# mágicos). Fuentes:
#   - Glucosa / HbA1c: American Diabetes Association (ADA), Standards of Care.
#   - Presión arterial: ACC/AHA 2017 Hypertension Guideline. Desde la v2 cuenta
#     tambien la diastolica: el objetivo de hipertension es >= 140/90, y con solo la
#     sistolica un 132/95 salia como estadio 1.
#   - Función renal: KDIGO 2024 (filtrado glomerular G1-G5, albuminuria A1-A3). Una
#     sola medición no basta para el diagnóstico: tiene que mantenerse 3 meses.
_NIVELES_PRESION = (
    ("presion_elevada", "presión elevada"),
    ("hipertension_estadio_1", "hipertensión estadio 1"),
    ("hipertension_estadio_2", "hipertensión estadio 2"),
    ("crisis_hipertensiva", "crisis hipertensiva"),
)
# (por debajo de, estadio, descripción): los estadios con filtrado < 60.
_ESTADIOS_FILTRADO = (
    (15, "G5", "fallo renal"),
    (30, "G4", "disminución grave"),
    (45, "G3b", "disminución moderada a grave"),
    (60, "G3a", "disminución leve a moderada"),
)


def compute_clinical_flags(glucose_mgdl: float, hba1c: float, systolic: float,
                           diastolic: float = 0.0, egfr: float = 0.0,
                           albumin_creatinine: float = 0.0) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []

    # --- Glucosa plasmática en ayunas (ADA) ---
    if glucose_mgdl >= 200:
        flags.append({"indicator": "glucose", "value": glucose_mgdl, "category": "diabetes",
                      "source": "ADA", "detail": "Glucosa ≥200 mg/dL: valor compatible con diabetes."})
    elif glucose_mgdl >= 126:
        flags.append({"indicator": "glucose", "value": glucose_mgdl, "category": "diabetes",
                      "source": "ADA", "detail": "Glucosa en ayuno ≥126 mg/dL: criterio de diabetes."})
    elif glucose_mgdl >= 100:
        flags.append({"indicator": "glucose", "value": glucose_mgdl, "category": "prediabetes",
                      "source": "ADA", "detail": "Glucosa en ayuno 100–125 mg/dL: rango de prediabetes."})

    # --- HbA1c (ADA) ---
    if hba1c >= 6.5:
        flags.append({"indicator": "hba1c", "value": hba1c, "category": "diabetes",
                      "source": "ADA", "detail": "HbA1c ≥6.5%: criterio de diabetes."})
    elif hba1c >= 5.7:
        flags.append({"indicator": "hba1c", "value": hba1c, "category": "prediabetes",
                      "source": "ADA", "detail": "HbA1c 5.7–6.4%: rango de prediabetes."})

    # --- Presión arterial (ACC/AHA 2017): manda la más alta de las dos ---
    nivel = -1
    if systolic >= 180 or diastolic >= 120:
        nivel = 3
    elif systolic >= 140 or diastolic >= 90:
        nivel = 2
    elif systolic >= 130 or diastolic >= 80:
        nivel = 1
    elif systolic >= 120:
        nivel = 0
    if nivel >= 0:
        lectura = "/".join(f"{v:g}" for v in (systolic, diastolic) if v) + " mmHg"
        categoria, nombre = _NIVELES_PRESION[nivel]
        flags.append({"indicator": "blood_pressure", "value": systolic or diastolic,
                      "category": categoria, "source": "ACC/AHA",
                      "detail": f"Presión {lectura}: {nombre}."})

    # --- Función renal (KDIGO): filtrado < 60 o albuminuria >= 30 mg/g ---
    if 0 < egfr < 60:
        estadio, nombre = next((e, n) for tope, e, n in _ESTADIOS_FILTRADO if egfr < tope)
        flags.append({"indicator": "egfr", "value": egfr, "category": f"filtrado_{estadio}",
                      "source": "KDIGO",
                      "detail": (f"Filtrado glomerular de {egfr:g} mL/min/1,73 m²: estadio "
                                 f"{estadio} ({nombre}). Por debajo de 60 es criterio de "
                                 f"enfermedad renal crónica si se mantiene 3 meses.")})
    if albumin_creatinine >= 30:
        categoria, nombre = (("A3", "gravemente aumentada") if albumin_creatinine > 300
                             else ("A2", "moderadamente aumentada"))
        flags.append({"indicator": "albuminuria", "value": albumin_creatinine,
                      "category": f"albuminuria_{categoria}", "source": "KDIGO",
                      "detail": (f"Albúmina/creatinina en orina de {albumin_creatinine:g} mg/g: "
                                 f"albuminuria {nombre} ({categoria}). Desde 30 es criterio de "
                                 f"enfermedad renal crónica si se mantiene 3 meses.")})

    return flags


# Variables que el modelo servido no usa (SIN_EFECTO). Pasa en los modos completos:
# en datos de un solo momento, quien ya esta diagnosticado suele estar tratado
# (antihipertensivos, estatinas) y su presion o su colesterol medidos ya no reflejan el
# riesgo. Las restricciones de monotonia impiden que el modelo aprenda la relacion al
# reves, y sin senal en el sentido clinico el efecto queda en cero. Callarlo seria
# otra forma de mentir: si el usuario las aporta, se dice.
_NOMBRES = {
    "ap_hi": "la presión sistólica", "ap_lo": "la presión diastólica",
    "total_cholesterol": "el colesterol total", "hdl_cholesterol": "el colesterol HDL",
    "hba1c_level": "la HbA1c", "egfr": "el filtrado glomerular",
    "albumin_creatinine_ratio": "la albúmina en orina", "waist_circumference": "la cintura",
    "alt": "la transaminasa ALT",
    "bmi": "el IMC",
}


def _enumerar(cosas: List[str]) -> str:
    return cosas[0] if len(cosas) == 1 else ", ".join(cosas[:-1]) + " y " + cosas[-1]


def _marcada(datos: Dict[str, Any], f: str) -> bool:
    """Una variable sin efecto 'cuenta' si el usuario la dio; una de si/no (una categoria
    del tabaco, un diagnostico), solo si la marco: el colesterol alto de la renal llega
    siempre, tambien con un "no", y avisar entonces de que no se refleja no tiene sentido."""
    crudo = _safe_get(datos, f)
    if _vacio(crudo):
        return False
    if _input_limits(f) == _BINARIAS:
        try:
            return _to_float(f, crudo) == 1
        except InvalidPayload:
            return False
    return True


# Diagnosticos previos (si/no) que pueden quedarse sin efecto, como el colesterol alto
# en la renal: quien lo tiene diagnosticado suele estar tratado (estatinas).
_DIAGNOSTICOS = {"hypertension": "hipertensión", "diabetes": "diabetes",
                 "high_cholesterol": "colesterol alto",
                 "heart_disease": "una enfermedad cardiovascular"}


def compute_sin_efecto_flags(clave: str, datos: Dict[str, Any]) -> List[Dict[str, Any]]:
    marcadas = [f for f in SIN_EFECTO.get(clave, []) if _marcada(datos, f)]
    flags: List[Dict[str, Any]] = []
    tabaco = [f for f in marcadas if f.startswith("smoking_history_")]
    if tabaco:
        flags.append({
            "indicator": "sin_efecto", "value": tabaco, "category": "no_reflejado_en_el_modelo",
            "source": "modelo",
            "detail": ("Indicaste que fumas o has fumado. El tabaco es un factor de riesgo, pero "
                       "esta estimación no lo refleja: en datos de un solo momento quien enferma "
                       "suele dejar de fumar, y el modelo no encuentra señal en el sentido clínico, "
                       "así que no le da peso."),
        })
    diagnosticos = [f for f in marcadas if f in _DIAGNOSTICOS]
    if diagnosticos:
        flags.append({
            "indicator": "sin_efecto", "value": diagnosticos, "category": "no_reflejado_en_el_modelo",
            "source": "modelo",
            # Sin "es un factor de riesgo": una cardiopatia previa no causa higado graso.
            "detail": (f"Indicaste un diagnóstico de {_enumerar([_DIAGNOSTICOS[f] for f in diagnosticos])}. "
                       f"Esta estimación no lo refleja: en los datos de entrenamiento el modelo no "
                       f"encuentra señal en el sentido clínico (a menudo porque quien ya está "
                       f"diagnosticado está en tratamiento), así que no le da peso."),
        })
    resto = [f for f in marcadas if f not in tabaco and f not in diagnosticos]
    if resto:
        texto = _enumerar([_NOMBRES.get(f, f) for f in resto])
        plural = len(resto) > 1
        flags.append({
            "indicator": "sin_efecto", "value": resto, "category": "no_reflejado_en_el_modelo",
            "source": "modelo",
            "detail": (f"{texto[0].upper() + texto[1:]} no {'cambian' if plural else 'cambia'} esta "
                       f"estimación: en los datos de entrenamiento, quien ya está diagnosticado "
                       f"suele estar en tratamiento, y el modelo no encuentra en "
                       f"{'ellos' if plural else 'ese dato'} señal en el sentido clínico. "
                       f"{'Se interpretan' if plural else 'Se interpreta'} aparte, con las guías."),
        })
    return flags


# ==========================================
# COBERTURA DE DATOS DE ENTRENAMIENTO (AUD-16)
# ==========================================
# Un modelo no avisa de que esta extrapolando: devuelve un numero igual de firme
# para una edad que vio 5000 veces que para una que no vio nunca. Esto NO toca la
# probabilidad (misma decision que la capa clinica en A4): se calcula aparte y se
# devuelve al lado del numero para que el usuario sepa cuanto fiarse.
SUPPORT: Dict[str, Dict[str, Dict[str, float]]] = {}

# Cuantos valores distintos hace falta para tratar una columna como continua: las
# dummies (0/1) no tienen "rango de soporte".
_MIN_VALORES_CONTINUA = 6


def _build_support_stats(clave: str) -> Dict[str, Dict[str, float]]:
    """Rango observado en el TRAIN real de cada feature continua del modelo (y del
    peso y la talla del simplificado). Se calcula una vez al arrancar."""
    df = _train_modo(clave)
    stats: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        if col == "target":
            continue
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.nunique() < _MIN_VALORES_CONTINUA:
            continue
        stats[col] = {
            "min": float(serie.min()),
            "max": float(serie.max()),
            "p1": float(serie.quantile(0.01)),
            "p99": float(serie.quantile(0.99)),
            "n": int(len(serie)),
        }
    return stats


def compute_support_warnings(clave: str, datos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Avisos por feature cuando el valor cae donde el modelo tiene pocos datos o
    ninguno. Solo mira las features que el modelo USA de verdad (las sin efecto no
    mueven la estimacion, asi que no hay extrapolacion que avisar)."""
    stats = SUPPORT.get(clave, {})
    if not stats:
        return []

    disease, _ = _partes(clave)
    topes = TOPCODED.get(disease, {})
    sin = set(SIN_EFECTO.get(clave, []))
    avisos: List[Dict[str, Any]] = []
    for feat in FEATURES.get(clave, []):
        rango = stats.get(feat)
        if not rango or feat in sin:
            continue
        crudo = _safe_get(datos, feat)
        if _vacio(crudo):
            continue
        try:
            valor = _to_float(feat, crudo)
        except InvalidPayload:
            continue

        lo, hi, p1, p99 = rango["min"], rango["max"], rango["p1"], rango["p99"]
        tope = topes.get(feat)
        if tope is not None and valor > tope:
            # Por encima del tope de NHANES SI hubo casos, pero codificados en el tope:
            # el modelo no los distingue de el y prolonga la tendencia aprendida.
            avisos.append({
                "feature": feat, "value": valor, "level": "sin_datos",
                "trained_range": [lo, hi], "topcoded": tope,
                "detail": (f"En estos datos todo el que pasa de {tope:g} figura como "
                           f"{tope:g}: el modelo sí vio casos así, pero no puede "
                           f"distinguirlos de {tope:g}. Por encima, la estimación prolonga "
                           f"la tendencia que aprendió (extrapolación)."),
            })
        elif valor < lo or valor > hi:
            avisos.append({
                "feature": feat, "value": valor, "level": "sin_datos",
                "trained_range": [lo, hi],
                "detail": (f"El modelo no vio ningún caso con este valor: en los datos de "
                           f"entrenamiento va de {lo:g} a {hi:g}. Aquí la estimación es una "
                           f"extrapolación."),
            })
        elif valor < p1 or valor > p99:
            avisos.append({
                "feature": feat, "value": valor, "level": "pocos_datos",
                "trained_range": [lo, hi], "common_range": [p1, p99],
                "detail": (f"Valor poco frecuente: el 98% de los datos de entrenamiento está "
                           f"entre {p1:g} y {p99:g}. La estimación aquí es menos fiable."),
            })
    return avisos


# ==========================================
# COHERENCIA ENTRE CAMPOS (auditoria 2026-08; presion: revision 2026-10)
# ==========================================
# Combinaciones que no pueden ser de una misma persona aunque cada dato caiga en su
# rango: una cintura que no cuadra con el IMC, una diastolica igual o mayor que la
# sistolica, o un IMC dado a mano que con ese peso implica una talla imposible. Las
# reglas estan en coherence.py, que comparte el generador de datos sinteticos. Igual
# que el resto de esta capa, NO toca la probabilidad: solo agrega un aviso.
def _check_coherencia(datos: Dict[str, Any], juzgar_peso: bool) -> List[Dict[str, Any]]:
    def _num(feat):
        crudo = _safe_get(datos, feat)
        if _vacio(crudo):
            return None
        try:
            v = _to_float(feat, crudo)
        except InvalidPayload:
            return None
        return v if v > 0 else None

    # El peso solo se juzga contra el IMC en el modo que pide los dos (el simplificado)
    # y si el IMC llego a mano: si salio del peso y la talla, cuadra por construccion.
    valores = {f: _num(f) for f in coherence.CAMPOS if f != "weight" or juzgar_peso}
    return [{"feature": a["feature"], "value": a["value"], "level": "incoherente",
             "detail": a["detail"]} for a in coherence.incoherencias(valores)]


# ==========================================
# BANDAS DE RIESGO (revision 2026-10)
# ==========================================
# "Bajo" es quedar por debajo de la media de los datos de entrenamiento y "alto", al
# menos el doble, con los tercios (33% / 66%) como tope. La regla y su porque estan
# en risk_banding.py, que comparten los scripts de entrenamiento; aqui solo se le da
# la prevalencia de cada enfermedad (la comparten sus dos modos). Igual que la capa
# clinica, la banda es una LECTURA de la probabilidad: no la modifica.
PREVALENCE: Dict[str, float] = {}  # enfermedad -> fraccion de positivos en el train real


def _build_prevalence(disease: str) -> Optional[float]:
    path = os.path.join(DATA_CURATED, disease, f"{disease}_train.csv")
    if not os.path.exists(path):
        return None
    target = pd.to_numeric(pd.read_csv(path, usecols=["target"])["target"], errors="coerce").dropna()
    return float(target.mean()) if len(target) else None


def risk_bands(disease: str) -> Dict[str, Any]:
    """Cortes de las bandas de una enfermedad: por debajo de `low_below` es bajo;
    desde `high_from`, alto."""
    return band_cutoffs(PREVALENCE.get(disease))


def risk_band(disease: str, prob: float) -> str:
    return band_of(prob, risk_bands(disease))


# ==========================================
# DEL PAYLOAD A LA FILA DEL MODELO
# ==========================================
def _derivar(modo: str, payload: Dict[str, Any]):
    """En el simplificado el usuario da su peso y su talla, no su IMC: se calcula aqui.
    Si llega el IMC (un cliente de la API que ya lo tiene), se usa tal cual. Devuelve
    (datos con el IMC, si el IMC vino dado)."""
    datos = dict(payload)
    bmi_dado = not _vacio(_safe_get(datos, "bmi"))
    entradas = M.DERIVADAS.get((modo, "bmi"))
    if not bmi_dado and entradas:
        peso, talla = (_safe_get(datos, e) for e in entradas)
        if not _vacio(peso) and not _vacio(talla):
            p = _to_valid_input("weight", peso)
            t = _to_valid_input("height", talla)
            imc = p / (t / 100) ** 2
            lo, hi = INPUT_LIMITS["bmi"]
            if not lo <= imc <= hi:
                raise InvalidPayload(
                    f"el peso ({p:g} kg) y la talla ({t:g} cm) dan un IMC de {imc:.1f}, "
                    f"fuera de lo físicamente posible ({lo:g}-{hi:g})")
            datos["bmi"] = round(imc, 2)
    return datos, bmi_dado


def _fila(clave: str, datos: Dict[str, Any]) -> np.ndarray:
    """Vector de features en el orden del modelo. Toda feature es obligatoria: la v1
    rellenaba con 0 las ausentes, y un IMC de 0 no es un paciente. Las unicas que
    pueden faltar son las que el modelo no usa (SIN_EFECTO), que toman la mediana del
    train: no cambian nada. Cada grupo (sexo, tabaco) tiene que traer una categoria y
    solo una, aunque el modelo no lea todas (el tabaco va contra "nunca")."""
    feats = FEATURES[clave]
    sin = set(SIN_EFECTO.get(clave, []))
    fila, faltan = [], []
    for f in feats:
        crudo = _safe_get(datos, f)
        if _vacio(crudo):
            if f in sin:
                fila.append(MEDIANAS.get(clave, {}).get(f, 0.0))
            else:
                fila.append(0.0)
                if not f.startswith(_PREFIJOS_ONE_HOT):
                    faltan.append(f)
            continue
        fila.append(_to_valid_input(f, crudo))  # no numerico o fuera de rango -> 400
    for grupo, opciones in M.CATEGORICAS.items():
        if not any(f.startswith(f"{grupo}_") for f in feats):
            continue
        claves = [f"{grupo}_{o}" for o in opciones]
        marcadas = sum(_to_valid_input(k, v) for k in claves
                       if not _vacio(v := _safe_get(datos, k)))
        if marcadas != 1:
            faltan.append(f"{_NOMBRE_GRUPO.get(grupo, grupo)} (una sola de: {', '.join(claves)})")
    if faltan:
        disease, modo = _partes(clave)
        pista = " (o weight y height)" if "bmi" in faltan and (modo, "bmi") in M.DERIVADAS else ""
        raise InvalidPayload(f"faltan datos del modo {modo}: {', '.join(faltan)}{pista}")
    return np.array([fila], dtype=float)


def _optional_clinical_value(datos: Dict[str, Any], key: str) -> float:
    """Dato clinico (glucosa, HbA1c, presion) o 0.0 si no viene. Pasa por la misma
    validacion que las features: un `1e999` salia como `Infinity` en clinical_flags,
    JSON invalido que rompe el JSON.parse del navegador (AUD-2)."""
    crudo = _safe_get(datos, key)
    if _vacio(crudo):
        return 0.0
    return _to_valid_input(key, crudo)


def _predict_proba(clave: str, X: np.ndarray):
    """Probabilidad de clase positiva: (cruda, calibrada). La calibrada aplica la
    isotonica centrada, un mapeo monotono: preserva el orden (AUC) y la monotonia
    clinica. SHAP explica SIEMPRE el modelo crudo."""
    raw = float(MODELS[clave].predict_proba(X)[0, 1])
    raw = min(max(raw, 0.0), 1.0)
    cal_obj = CALIBRATORS.get(clave)
    cal = float(cal_obj.predict([raw])[0]) if cal_obj is not None else raw
    return raw, min(max(cal, 0.0), 1.0)


def support_note(avisos: List[Dict[str, Any]]) -> str:
    if not avisos:
        return "Todos los valores caen dentro del rango con datos de entrenamiento."
    if any(a["level"] == "sin_datos" for a in avisos):
        return ("Alguno de los datos queda fuera del rango que el modelo llegó a ver, "
                "así que este resultado es una extrapolación: tómalo con reservas.")
    if any(a["level"] == "incoherente" for a in avisos):
        return ("Alguno de los datos ingresados no es coherente con el resto (p.ej. la "
                "cintura y el IMC no cuadran entre sí): tómalo con las mismas reservas que "
                "una extrapolación.")
    return ("Alguno de los datos cae en una zona con pocos ejemplos de entrenamiento, "
            "así que la estimación es menos fiable de lo habitual.")

# ==========================================
# FUNCIONES AUXILIARES PARA DATOS SINTÉTICOS / REALES
# ==========================================
@lru_cache(maxsize=16)
def _read_csv_cached(csv_path: str) -> pd.DataFrame:
    """Los CSV del laboratorio no cambian con el servidor en marcha (el pipeline los
    regenera offline y un reinicio los recarga). Antes /sample, /synthetic y
    /distribution releian el CSV entero en CADA peticion: una forma barata de cargar
    el servidor. OJO: el DataFrame es compartido, quien lo use no debe mutarlo."""
    return pd.read_csv(csv_path)


def _synthetic_files(disease: str) -> List[str]:
    """Sinteticos de una enfermedad, CTGAN primero. Ordenados: glob no garantiza
    orden y con varios archivos se servia uno u otro segun el sistema de archivos."""
    base_dir = os.path.join(DATA_CURATED, disease)
    return (sorted(glob.glob(os.path.join(base_dir, f"{disease}_synthetic_ctgan*.csv")))
            or sorted(glob.glob(os.path.join(base_dir, f"{disease}_synthetic*.csv"))))


def _columnas_ficha(disease: str) -> List[str]:
    """Lo que trae una ficha del laboratorio, real o sintetica: las mismas columnas en
    las dos (clave para el juego 'real o sintetico'). Sin el objetivo."""
    return [c for c in M.columnas_laboratorio(disease) if c != "target"]


def _sample_row(df: pd.DataFrame, disease: str, source_type: str):
    """Muestrea 1 fila completa con las columnas de la ficha, normaliza tipos numpy y
    marca _source_type. Mismo formato para datos reales y sinteticos."""
    try:
        cols = [c for c in _columnas_ficha(disease) if c in df.columns]
        sample = df[cols].dropna().sample(1).iloc[0].to_dict()
        sample = {k: _json_safe(v) for k, v in sample.items()}
        sample["_source_type"] = source_type
        return sample
    except Exception as e:
        print(f"Error muestreando ({disease}, {source_type}): {e}")
        return None


def get_real_sample(disease):
    """Una fila REAL aleatoria del train (data_curated/<disease>/<disease>_train.csv)."""
    csv_path = os.path.join(DATA_CURATED, disease, f"{disease}_train.csv")
    if not os.path.exists(csv_path):
        return None
    return _sample_row(_read_csv_cached(csv_path), disease, "real")


def get_random_sample(disease):
    """Una fila SINTETICA (CTGAN primero); sin sinteticos, una real marcada como real."""
    found_files = _synthetic_files(disease)
    if not found_files:
        print(f"⚠️ No se hallaron sintéticos para {disease}. Usando datos reales.")
        return get_real_sample(disease)
    return _sample_row(_read_csv_cached(found_files[0]), disease, "synthetic")


# /health esta exento del rate limiting (AUD-4: un 429 marcaria el deploy como caido
# ante un monitor de uptime), asi que no puede costar una consulta a la BD por
# peticion: el resultado se reutiliza unos segundos. Un monitor pincha cada 30-60 s y
# sigue viendo el estado real; un bucle contra /health ya no llega a la BD.
_HEALTH_DB_TTL_S = 5.0
_health_db = {"checked_at": None, "ok": False}


def _db_ok_cached() -> bool:
    ahora = time.monotonic()
    if _health_db["checked_at"] is None or ahora - _health_db["checked_at"] > _HEALTH_DB_TTL_S:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            _health_db["ok"] = True
        except Exception as e:
            _health_db["ok"] = False
            print(f"[WARN] /health: la BD no responde: {e}")
        _health_db["checked_at"] = ahora
    return _health_db["ok"]


@app.get("/health")
@limiter.exempt
def health():
    """Liveness + comprobacion REAL de la BD y de los modelos: sin ellos la app
    levanta igual, pero /predict responde 500."""
    db_ok = _db_ok_cached()
    faltan = _modelos_faltantes()
    ok = db_ok and not faltan
    payload = {
        "status": "ok" if ok else "degraded",
        "database": engine.url.get_backend_name(),   # sqlite | postgresql | ...
        "database_ok": db_ok,
        "models_loaded": sorted(MODELS.keys()),
        "models_missing": faltan,
    }
    return jsonify(payload), (200 if ok else 503)


def _enfermedad_y_modo(disease: str):
    """(enfermedad, modo, None) o (None, None, respuesta de error). El modo va en
    ?mode=simplificado|completo; sin el, el simplificado."""
    disease = disease.lower()
    if disease not in ENFERMEDADES:
        return None, None, (jsonify({"error": "unknown disease"}), 404)
    modo = (request.args.get("mode") or MODO_POR_DEFECTO).lower()
    if modo not in MODOS:
        return None, None, (jsonify({"error": f"modo desconocido: usa {' o '.join(MODOS)}"}), 400)
    return disease, modo, None


def _leer_json(nombre: str):
    """Cuerpo JSON como dict, o (None, respuesta de error)."""
    try:
        cuerpo = request.get_json(force=True) or {}
    except RequestEntityTooLarge:
        raise  # cuerpo por encima de MAX_CONTENT_LENGTH -> 413 (AUD-9)
    except Exception:
        return None, (jsonify({"error": "invalid JSON"}), 400)
    if not isinstance(cuerpo, dict):
        return None, (jsonify({"error": f"{nombre} debe ser un objeto JSON"}), 400)
    return cuerpo, None


@app.get("/metrics/<disease>")
def get_metrics(disease: str):
    disease, modo, error = _enfermedad_y_modo(disease)
    if error:
        return error
    mpath = _paths(_clave(disease, modo))["metrics"]
    if not os.path.exists(mpath):
        return jsonify({"error": "metrics not found"}), 404
    with open(mpath, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.get("/config/<disease>")
def get_config(disease: str):
    disease, modo, error = _enfermedad_y_modo(disease)
    if error:
        return error
    clave = _clave(disease, modo)
    if clave not in FEATURES:
        return jsonify({"error": "features not found"}), 404
    feats = FEATURES[clave]
    # Lo que pide el formulario: las features, salvo las que la API calcula (el IMC del
    # simplificado sale del peso y la talla).
    derivadas = {f: list(M.DERIVADAS[(modo, f)]) for f in feats if (modo, f) in M.DERIVADAS}
    # Los grupos (sexo, tabaco) son UNA pregunta: van por su nombre, y sus opciones en
    # `categoricals`.
    entradas: List[str] = []
    for f in feats:
        grupo = next((g for g in M.CATEGORICAS if f.startswith(f"{g}_")), None)
        for e in ([grupo] if grupo else derivadas.get(f, [f])):
            if e not in entradas:
                entradas.append(e)
    # Opcionales que no son del modelo pero si de la capa clinica.
    clinicas = _clinicas_del_formulario(disease, feats)
    ranges = {f: _input_limits(f) for f in dict.fromkeys(entradas + feats + clinicas)
              if f not in M.CATEGORICAS and _input_limits(f)}

    return jsonify({
        "disease": disease,
        "mode": modo,
        "modes": MODOS,
        "features": feats,
        # Campos del formulario (con el peso y la talla en vez del IMC si toca).
        "inputs": entradas,
        "derived": derivadas,
        "optional_features": clinicas,
        # Opcionales que NO cambian la estimacion: solo los lee la capa clinica.
        "clinical_inputs": clinicas,
        # Las que definen la enfermedad: se piden en el completo, las lee la guia.
        "defining_inputs": list(M.DEFINITORIAS[disease]),
        # Las que el modelo acepta pero no usa (restricciones de monotonia).
        "not_used": SIN_EFECTO.get(clave, []),
        "ranges": ranges,
        # AUD-16: rango REAL de los datos de entrenamiento, para que el front pueda
        # avisar de que fuera de ahi el modelo extrapola. `ranges` de arriba son los
        # limites que se aceptan, que es otra cosa.
        "feature_support": SUPPORT.get(clave, {}),
        # Features con tope en los datos (NHANES: edad 80 = 80 o mas).
        "topcoded": TOPCODED.get(disease, {}),
        # Cortes de las bandas bajo / moderado / alto de esta enfermedad.
        "risk_bands": risk_bands(disease),
        "categoricals": {g: list(opciones) for g, opciones in M.CATEGORICAS.items()
                         if g in entradas},
    })


@app.post("/predict/<disease>")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict(disease: str):
    disease, modo, error = _enfermedad_y_modo(disease)
    if error:
        return error
    clave = _clave(disease, modo)
    if clave not in MODELS or clave not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 500

    payload, error = _leer_json("el cuerpo")
    if error:
        return error

    feats = FEATURES[clave]
    try:
        datos, bmi_dado = _derivar(modo, payload)
        X = _fila(clave, datos)
        clinico = {k: _optional_clinical_value(datos, k) for k in ENTRADAS_CLINICAS}
    except InvalidPayload as e:
        return jsonify({"error": str(e)}), 400

    # 1. Predicción del modelo (cruda) + calibración isotónica centrada.
    raw_prob, prob = _predict_proba(clave, X)

    # >>> SHAP START
    top_features = []
    shap_vals = _shap_fila(clave, X)
    if shap_vals is not None:
        x_row = X.reshape(-1)
        # El sexo no se explica (no es un factor sobre el que actuar); las variables sin
        # efecto tampoco, porque su SHAP es 0.
        sin = set(SIN_EFECTO.get(clave, []))
        permitidos = [i for i, f in enumerate(feats) if not f.startswith("gender_") and f not in sin]
        for i in sorted(permitidos, key=lambda i: abs(shap_vals[i]), reverse=True)[:5]:
            top_features.append({
                "feature": feats[i],
                "value": float(x_row[i]),
                "shap": float(shap_vals[i]),
                "abs_shap": float(abs(shap_vals[i])),
            })
    # >>> SHAP END

    # La clase es la del MODELO (su salida cruda >= 0,5, lo mismo que pipeline.predict):
    # es la regla que evalua train_models.py y que /metricas publica como sensibilidad.
    pred_class = 1 if raw_prob >= 0.5 else 0

    clinical_flags = compute_clinical_flags(clinico["blood_glucose_level"], clinico["hba1c_level"],
                                            clinico["ap_hi"], clinico["ap_lo"], clinico["egfr"],
                                            clinico["albumin_creatinine_ratio"])
    clinical_flags += compute_sin_efecto_flags(clave, datos)
    clinical_note = " ".join(f["detail"] for f in clinical_flags) or \
        "Sin indicadores clínicos por encima de umbrales de referencia."

    # AUD-16: igual que la capa clínica, se calcula APARTE y no toca la probabilidad.
    support_warnings = compute_support_warnings(clave, datos)
    # En el completo el peso no es dato del modo. El laboratorio manda la ficha entera,
    # con el peso DECLARADO junto al IMC MEDIDO, y cruzar dos mediciones distintas daba
    # una talla implicita de menos de 1,30 m a 9 de cada 9.500 personas reales.
    support_warnings += _check_coherencia(datos, bmi_dado and (modo, "bmi") in M.DERIVADAS)

    # Los pacientes del GAN que evalua el laboratorio (Proyecto.jsx) no son simulaciones
    # de nadie: llegan con ?source=synthetic (como en /sample) y, igual que /whatif, no
    # se registran.
    if (request.args.get("source") or "").lower() != "synthetic":
        log_prediction_to_db(
            disease, _loggable_payload(clave, datos), pred_class, prob,
            model_name=MODEL_NAMES.get(clave),
            clinical_note=clinical_note,
            top_features=top_features,
            session_id=_clean_session_id(request.headers.get("X-Session-Id")),
            mode=modo,
        )

    return jsonify({
        "disease": disease,
        "mode": modo,
        "model": MODEL_NAMES.get(clave),
        "probability": prob,
        "raw_model_probability": raw_prob,
        "calibrated": clave in CALIBRATORS,
        # Lectura de `probability` frente a la media de la enfermedad: low / mid / high.
        "risk_band": risk_band(disease, prob),
        "risk_bands": risk_bands(disease),
        "prediction": pred_class,
        # El IMC con el que se calculo (en el simplificado sale del peso y la talla).
        "bmi": datos.get("bmi"),
        "top_features": top_features,
        "clinical_flags": clinical_flags,
        "clinical_note": clinical_note,
        "not_used": SIN_EFECTO.get(clave, []),
        "support_warnings": support_warnings,
        "support_note": support_note(support_warnings),
        "explain_note": ("La probabilidad mostrada es la salida del modelo calibrada (isotónica "
                         "centrada); `raw_model_probability` es la salida cruda, la que explican los "
                         "valores SHAP. Estima la enfermedad total: diagnosticada o detectada por "
                         "análisis o medición. Los indicadores clínicos (ADA, ACC/AHA, KDIGO) y los avisos "
                         "de cobertura de datos (`support_warnings`) se muestran aparte como "
                         "referencia y NO modifican la probabilidad."),
    })


def _barribles(clave: str) -> List[str]:
    """Variables que el what-if puede barrer: las continuas que el modelo usa de verdad
    (barrer una que no usa daba una recta sin sentido) y el peso del simplificado."""
    disease, modo = _partes(clave)
    sin = set(SIN_EFECTO.get(clave, []))
    continuas = [f for f in FEATURES[clave]
                 if f not in sin and not f.startswith(_PREFIJOS_ONE_HOT)
                 and _input_limits(f) != _BINARIAS]
    return continuas + [e for e in M.DERIVADAS.get((modo, "bmi"), ()) if e == "weight"]


@app.post("/whatif/<disease>")
@limiter.limit(RATE_LIMIT_PREDICT)
def whatif(disease: str):
    """Análisis contrafactual: fija un caso base y barre UNA variable sobre un rango,
    devolviendo la curva de riesgo (probabilidad calibrada). NO se registra en la BD
    (no ensucia la analítica del admin) y no calcula SHAP. En el simplificado, barrer
    el peso mueve el IMC a la talla del caso base, y barrer el IMC mueve el peso."""
    disease, modo, error = _enfermedad_y_modo(disease)
    if error:
        return error
    clave = _clave(disease, modo)
    if clave not in MODELS or clave not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 500

    body, error = _leer_json("el cuerpo")
    if error:
        return error
    feature = body.get("feature")
    if not feature:
        return jsonify({"error": "missing 'feature'"}), 400
    try:
        vmin, vmax = float(body["min"]), float(body["max"])
        steps = int(body.get("steps", 25))
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "invalid min/max/steps"}), 400
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return jsonify({"error": "'max' must be greater than 'min'"}), 400
    steps = max(2, min(steps, 100))

    base = body.get("base") or {}
    if not isinstance(base, dict):
        return jsonify({"error": "'base' debe ser un objeto JSON"}), 400
    if feature not in _barribles(clave):
        return jsonify({"error": f"'{feature}' no es una variable que el modelo de {disease} "
                                 f"({modo}) use: daria una recta"}), 400

    # Peso <-> IMC a la talla del caso base (simplificado): sin talla no hay acople.
    talla_m = None
    if feature in ("weight", "bmi") and (modo, "bmi") in M.DERIVADAS:
        try:
            crudo = _safe_get(base, "height")
            talla_m = None if _vacio(crudo) else _to_valid_input("height", crudo) / 100
        except InvalidPayload as e:
            return jsonify({"error": str(e)}), 400
        if feature == "weight" and talla_m is None:
            return jsonify({"error": "barrer el peso necesita la talla del caso base"}), 400

    curve = []
    try:
        for i in range(steps):
            v = vmin + (vmax - vmin) * i / (steps - 1)
            datos = dict(base)
            punto = {"value": round(v, 2)}
            if feature == "weight":
                # Topado a los limites fisicos: un barrido ancho sobre una talla extrema
                # se saldria de ellos y un valor DERIVADO no debe tumbar la curva.
                lo, hi = INPUT_LIMITS["bmi"]
                datos["weight"] = v
                datos["bmi"] = min(max(v / talla_m ** 2, lo), hi)
                punto["coupled_value"] = round(datos["bmi"], 2)
            elif feature == "bmi":
                datos["bmi"] = v
                if talla_m is not None:
                    punto["coupled_value"] = round(v * talla_m ** 2, 2)
            else:
                datos[feature] = v
            datos, _ = _derivar(modo, datos)
            raw, cal = _predict_proba(clave, _fila(clave, datos))
            curve.append({**punto, "probability": cal, "raw_probability": raw})
    except InvalidPayload as e:
        return jsonify({"error": str(e)}), 400

    rango = SUPPORT.get(clave, {}).get(feature)
    acoplada = {"weight": "bmi", "bmi": "weight"}.get(feature) if talla_m is not None else None
    return jsonify({
        "disease": disease,
        "mode": modo,
        "feature": feature,
        "model": MODEL_NAMES.get(clave),
        "calibrated": clave in CALIBRATORS,
        "curve": curve,
        # AUD-16: rango de la variable barrida en el train real: fuera de aqui la curva
        # se aplana porque no hay datos, no porque el riesgo deje de subir.
        "supported_range": [rango["min"], rango["max"]] if rango else None,
        # NHANES: por encima de este tope si hubo casos, pero codificados en el tope.
        "topcoded_at": TOPCODED.get(disease, {}).get(feature),
        # Variable que se movio junto a la barrida para no cambiar la talla del caso
        # base (cada punto trae su `coupled_value`); null si el barrido fue de una sola.
        "coupled": ({"feature": acoplada, "height_m": round(talla_m, 2)} if acoplada else None),
    })


@app.get("/synthetic/<disease>")
def get_synthetic(disease):
    disease = disease.lower()
    if disease not in ENFERMEDADES:
        return jsonify({"error": "disease not supported"}), 404
    sample = get_random_sample(disease)
    if not sample:
        return jsonify({"error": "could not generate synthetic data"}), 500
    return jsonify(sample)


@app.get("/sample/<disease>")
def get_sample(disease):
    """Una ficha de paciente del origen pedido: ?source=real|synthetic
    (default synthetic). Alimenta el juego 'real vs sintético'."""
    disease = disease.lower()
    if disease not in ENFERMEDADES:
        return jsonify({"error": "disease not supported"}), 404
    source = (request.args.get("source") or "synthetic").lower()
    sample = get_real_sample(disease) if source == "real" else get_random_sample(disease)
    if not sample:
        return jsonify({"error": "could not get sample"}), 500
    return jsonify(sample)


@app.get("/distribution/<disease>")
def get_distribution(disease):
    """Histograma comparado real vs sintético de una variable numérica.
    ?feature=<col>&bins=<n>. Devuelve proporciones (cada serie suma ~100%) sobre
    bins COMUNES, para comparar la *forma* aunque difiera el tamaño de muestra."""
    disease = disease.lower()
    if disease not in ENFERMEDADES:
        return jsonify({"error": "disease not supported"}), 404

    feature = request.args.get("feature")
    if not feature:
        return jsonify({"error": "feature required"}), 400

    try:
        nbins = max(5, min(40, int(request.args.get("bins", 18))))
    except (TypeError, ValueError):
        nbins = 18

    real_path = os.path.join(DATA_CURATED, disease, f"{disease}_train.csv")
    synth_files = _synthetic_files(disease)
    if not os.path.exists(real_path) or not synth_files:
        return jsonify({"error": "data not available"}), 500

    try:
        real = _read_csv_cached(real_path)
        synth = _read_csv_cached(synth_files[0])
        if feature not in _columnas_ficha(disease) or feature not in real.columns \
                or feature not in synth.columns:
            return jsonify({"error": "feature not found"}), 400

        r = pd.to_numeric(real[feature], errors="coerce").dropna()
        s = pd.to_numeric(synth[feature], errors="coerce").dropna()
        if r.empty or s.empty:
            return jsonify({"error": "no numeric data"}), 400

        lo = float(min(r.min(), s.min()))
        hi = float(max(r.max(), s.max()))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, nbins + 1)
        r_counts, _ = np.histogram(r, bins=edges)
        s_counts, _ = np.histogram(s, bins=edges)
        r_sum = r_counts.sum() or 1
        s_sum = s_counts.sum() or 1

        bins = []
        for i in range(len(edges) - 1):
            bins.append({
                "bin": round((edges[i] + edges[i + 1]) / 2, 1),
                "real": round(float(r_counts[i] / r_sum * 100), 2),
                "synthetic": round(float(s_counts[i] / s_sum * 100), 2),
            })

        return jsonify({
            "feature": feature,
            "real_n": int(r.shape[0]),
            "synthetic_n": int(s.shape[0]),
            "bins": bins,
        })
    except Exception as e:
        print(f"Error en distribution ({disease}/{feature}): {e}")
        return jsonify({"error": "could not compute distribution"}), 500


_QUALITY_CACHE = {}


@app.get("/synthetic_quality/<disease>")
def get_synthetic_quality(disease):
    """Calidad del sintetico. Sirve el informe PRECOMPUTADO por el pipeline
    (build_quality_reports.py); solo lo recalcula si falta y hay sdmetrics
    instalado, porque en produccion no lo hay (arrastra torch, ~479 MB)."""
    disease = disease.lower()
    if disease not in ENFERMEDADES:
        return jsonify({"error": "disease not supported"}), 404
    if disease not in _QUALITY_CACHE:
        res = sq.load_precomputed(disease) or sq.compute(disease)
        if res is None:
            return jsonify({"error": "data not available"}), 500
        _QUALITY_CACHE[disease] = res
    return jsonify(_QUALITY_CACHE[disease])


# ==========================================
# PANEL ADMIN (dev-only, anónimo, server-side — A3)
# ==========================================
# Fuerza bruta del ADMIN_TOKEN (auditoria 2026-09). Los cuatro endpoints comprueban
# el token y contestan 401 si no cuadra, asi que cualquiera sirve para probarlo: con
# el limite estricto solo en /admin/verify salian 10 + 3 x 60 = 190 intentos por
# minuto. Ahora los fallos comparten una cubeta por IP. Solo descuentan los 401 (el
# uso con el token bueno no la gasta) y, agotada, todo el panel responde 429 tambien
# al token bueno: si no, el acierto se distinguiria. El descuento llega al terminar
# la peticion, asi que N peticiones simultaneas cuelan hasta N-1 intentos de mas.
_intentos_token_admin = limiter.shared_limit(
    RATE_LIMIT_ADMIN_VERIFY, scope="admin_token",
    deduct_when=lambda resp: resp.status_code == 401,
)


@app.get("/admin/verify")
@limiter.limit(RATE_LIMIT_ADMIN)
@_intentos_token_admin
@require_admin
def admin_verify():
    """El frontend lo usa para validar el token antes de mostrar el dashboard."""
    return jsonify({"ok": True})


def _parse_date_range():
    """Lee ?from=YYYY-MM-DD&to=YYYY-MM-DD de la query. Devuelve (dt_from, dt_to_excl)
    donde dt_to_excl es exclusivo (fin del día 'to'). Cualquiera puede ser None."""
    def _parse(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    dt_from = _parse(request.args.get("from"))
    dt_to = _parse(request.args.get("to"))
    dt_to_excl = (dt_to + timedelta(days=1)) if dt_to else None
    return dt_from, dt_to_excl


def _query_predictions_filtered(s):
    """Query base del admin: solo enfermedades servidas + filtro de rango de fechas."""
    q = s.query(Prediction).filter(Prediction.disease.in_(ENFERMEDADES))
    dt_from, dt_to_excl = _parse_date_range()
    if dt_from is not None:
        q = q.filter(Prediction.timestamp >= dt_from)
    if dt_to_excl is not None:
        q = q.filter(Prediction.timestamp < dt_to_excl)
    return q


@app.get("/admin/stats")
@limiter.limit(RATE_LIMIT_ADMIN)
@_intentos_token_admin
@require_admin
def admin_stats():
    """Analítica de uso AGREGADA y anónima (sin datos personales). Solo cuenta las
    enfermedades servidas; ignora filas viejas de enfermedades retiradas.
    Acepta ?from=&to= (YYYY-MM-DD) para acotar el rango. La agregación se hace en
    Python (probabilidades, horas, top-features) para ser agnóstica al motor SQL."""
    supported = list(ENFERMEDADES)
    with SessionLocal() as s:
        rows = _query_predictions_filtered(s).all()

    total = len(rows)
    distinct_sessions = len({r.session_id for r in rows if r.session_id})

    # --- Por enfermedad: conteo, prob media, positivos, tasa+ ---
    # --- Histograma de probabilidad por enfermedad (10 bins 0..1) ---
    # --- Frecuencia de features SHAP en el top ---
    N_BINS = 10
    by_d = {d: {"count": 0, "sum_prob": 0.0, "positives": 0,
                "hist": [0] * N_BINS} for d in supported}
    hourly = [0] * 24
    feat_freq: Dict[str, Dict[str, float]] = {}

    for r in rows:
        d = r.disease
        if d not in by_d:
            continue
        b = by_d[d]
        b["count"] += 1
        p = float(r.probability or 0.0)
        b["sum_prob"] += p
        b["positives"] += int(r.prediction or 0)
        idx = min(int(p * N_BINS), N_BINS - 1)
        b["hist"][idx] += 1
        if r.timestamp:
            hourly[r.timestamp.hour] += 1
        tf = _safe_json_loads(r.top_features)
        if isinstance(tf, list):
            for item in tf:
                if not isinstance(item, dict):
                    continue
                name = item.get("feature")
                if not name or str(name).startswith("gender_"):
                    continue
                agg = feat_freq.setdefault(name, {"count": 0, "sum_abs_shap": 0.0})
                agg["count"] += 1
                agg["sum_abs_shap"] += abs(float(item.get("shap", 0.0)))

    by_disease = []
    prob_histogram = {}
    for d in supported:
        b = by_d[d]
        n = b["count"]
        by_disease.append({
            "disease": d,
            "count": n,
            "avg_probability": round(b["sum_prob"] / n, 4) if n else 0.0,
            "positives": b["positives"],
            "positive_rate": round(b["positives"] / n, 4) if n else 0.0,
            "model": " / ".join(f"{m}: {MODEL_NAMES.get(_clave(d, m), '?')}" for m in MODOS),
            "models": {m: MODEL_NAMES.get(_clave(d, m)) for m in MODOS},
        })
        prob_histogram[d] = b["hist"]

    top_features = sorted(
        ({"feature": k, "count": v["count"],
          "avg_abs_shap": round(v["sum_abs_shap"] / v["count"], 4) if v["count"] else 0.0}
         for k, v in feat_freq.items()),
        key=lambda x: x["count"], reverse=True,
    )[:10]

    # --- Timeline diario (en Python, agnóstico al motor) ---
    daily: Dict[str, int] = {}
    for r in rows:
        if r.timestamp:
            key = r.timestamp.strftime("%Y-%m-%d")
            daily[key] = daily.get(key, 0) + 1
    timeline = [{"day": k, "count": daily[k]} for k in sorted(daily)]

    return jsonify({
        "total": total,
        "distinct_sessions": distinct_sessions,
        "by_disease": by_disease,
        "timeline": timeline,
        "prob_histogram": prob_histogram,
        "prob_bins": N_BINS,
        "hourly": hourly,
        "top_features": top_features,
    })


@app.get("/admin/predictions")
@limiter.limit(RATE_LIMIT_ADMIN)
@_intentos_token_admin
@require_admin
def admin_predictions():
    """Lista las simulaciones más recientes (server-side, anónimas)."""
    try:
        limit = int(request.args.get("limit", 50))
    except (ValueError, TypeError):
        limit = 50
    # Un limit <= 0 caia tal cual en la query, y SQLite lee LIMIT -1 como "sin
    # limite": ?limit=-1 devolvia la tabla entera saltandose el tope de 500.
    if limit < 1:
        limit = 50
    limit = min(limit, 500)
    disease = request.args.get("disease")

    with SessionLocal() as s:
        # Solo enfermedades servidas hoy (oculta filas viejas de `obesidad`, etc.)
        # + filtro opcional de rango de fechas (?from=&to=).
        q = _query_predictions_filtered(s).order_by(Prediction.id.desc())
        if disease:
            q = q.filter(Prediction.disease == disease.lower())
        rows = q.limit(limit).all()
        items = [{
            "id": r.id,
            "disease": r.disease,
            "prediction": r.prediction,
            "probability": r.probability,
            "model": r.model_name,
            "mode": r.mode,
            "clinical_note": r.clinical_note,
            "session_id": r.session_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "input_data": _safe_json_loads(r.input_data),
            "top_features": _safe_json_loads(r.top_features),
        } for r in rows]

    return jsonify({"count": len(items), "items": items})


@app.get("/admin/export.csv")
@limiter.limit(RATE_LIMIT_ADMIN)
@_intentos_token_admin
@require_admin
def admin_export_csv():
    """Exporta las simulaciones (anónimas) como CSV server-side. Respeta el filtro
    de rango de fechas (?from=&to=) y el de enfermedad (?disease=)."""
    disease = request.args.get("disease")
    with SessionLocal() as s:
        q = _query_predictions_filtered(s).order_by(Prediction.id.desc())
        if disease:
            q = q.filter(Prediction.disease == disease.lower())
        rows = q.all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "disease", "mode", "model", "prediction",
                     "probability", "session_id", "clinical_note", "input_data"])
    for r in rows:
        writer.writerow([
            r.id,
            r.timestamp.isoformat() if r.timestamp else "",
            _csv_safe(r.disease),
            _csv_safe(r.mode or ""),
            _csv_safe(r.model_name or ""),
            r.prediction,
            r.probability,
            _csv_safe(r.session_id or ""),
            _csv_safe((r.clinical_note or "").replace("\n", " ")),
            _csv_safe(r.input_data or ""),
        ])
    csv_data = buf.getvalue()
    return app.response_class(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predisana_simulaciones.csv"},
    )


# Inicialización global (se ejecuta siempre)
_load_all()
init_db()

# Inicialización local
if __name__ == "__main__":
    # debug=True expone el debugger interactivo de Werkzeug (traceback + consola)
    # a toda la red al escuchar en 0.0.0.0. Ahora es opt-in por env var.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes", "on")
    if not debug:
        print("[info] debug OFF (sin recarga automatica). Para activarlo: "
              "$env:FLASK_DEBUG=\"1\"; python app.py")
    app.run(host="0.0.0.0", port=8000, debug=debug)