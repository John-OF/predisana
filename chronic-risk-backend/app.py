import json
import os
import sys
import csv
import io
import re
import math
import hmac
from datetime import datetime, timedelta
from functools import wraps
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
from flask_cors import CORS
from joblib import load

from sqlalchemy import (
    create_engine, inspect, text, func,
    Column, Integer, String, Float, Text as SAText, DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import synthetic_quality as sq

EXPLAINERS: Dict[str, Any] = {}

app = Flask(__name__)
CORS(app)
# AUD-9: los payloads legitimos son de unos cientos de bytes (un puñado de
# features numericas). Sin tope, cualquiera puede mandar un cuerpo gigante y
# obligar al servidor a bufferearlo entero.
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # 256 KB


@app.errorhandler(413)
def _cuerpo_demasiado_grande(_e):
    return jsonify({"error": "cuerpo demasiado grande"}), 413

BASE_MODELS = "models"
DB_NAME = "medical_history.db"  # <--- Nombre de la Base de Datos
# Capa de datos agnóstica al motor (A3): SQLite en dev, Postgres en prod (#7),
# mismo código. Se controla con la env var DATABASE_URL.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_NAME}")
# Token del panel admin dev-only (A3). Si no está seteado, el admin queda
# deshabilitado (los endpoints /admin/* responden 503). NO es auth de usuario:
# los usuarios nunca se loguean, las simulaciones son anónimas.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

# Archivos por enfermedad
FILES = {
    "diabetes": {
        "pipeline": os.path.join(BASE_MODELS, "diabetes_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "diabetes_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "diabetes_metrics.json"),
    },
    "hipertension": {
        "pipeline": os.path.join(BASE_MODELS, "hipertension_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "hipertension_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "hipertension_metrics.json"),
    },
    "cardiovascular": {
        "pipeline": os.path.join(BASE_MODELS, "cardiovascular_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "cardiovascular_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "cardiovascular_metrics.json"),
    },
}

MODELS: Dict[str, Any] = {}
FEATURES: Dict[str, List[str]] = {}
MODEL_NAMES: Dict[str, str] = {}  # enfermedad -> modelo ganador (A5), desde _metrics.json
CALIBRATORS: Dict[str, Any] = {}  # enfermedad -> isotonica que calibra predict_proba

# Modelo HIBRIDO de diabetes (NHANES): además del modelo base servido por defecto
# ("diabetes", solo features respondibles), hay una VARIANTE con glucosa que se
# sirve cuando el usuario la ingresa. Se cargan como claves extra en los dicts de
# arriba, pero NO son enfermedades servibles por sí solas (no van en FILES ni en la
# UI). Mapa variante -> enfermedad base (para datos/UI/admin).
EXTRA_MODEL_KEYS = ["diabetes_glucosa"]
VARIANT_BASE = {"diabetes_glucosa": "diabetes"}
# Feature extra que activa cada variante de diabetes (si el usuario la aporta).
DIABETES_GLUCOSE_KEY = "diabetes_glucosa"


def _model_paths(key: str) -> Dict[str, str]:
    return {
        "pipeline": os.path.join(BASE_MODELS, f"{key}_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, f"{key}_features.json"),
        "metrics":  os.path.join(BASE_MODELS, f"{key}_metrics.json"),
    }


def _data_disease(key: str) -> str:
    """Enfermedad base a la que pertenece una clave de modelo (para carpetas de
    datos). Una variante como 'diabetes_glucosa' usa los datos de 'diabetes'."""
    return VARIANT_BASE.get(key, key)

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
                         top_features=None, session_id=None):
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


# Claves que se persisten aunque no sean features del modelo servido: alimentan
# la capa clinica (ADA) y son utiles para leer la simulacion en el admin.
_LOG_EXTRA_KEYS = ("glucose", "blood_glucose_level", "hba1c_level")


def _loggable_payload(key: str, payload: Dict[str, Any]) -> Dict[str, float]:
    """Recorta el payload a lo que el modelo (o la capa clinica) usa de verdad.
    Antes se guardaba el JSON entero tal cual: claves arbitrarias del cliente
    engordando la BD sin aportar nada, y texto libre en una tabla que se exporta."""
    permitidas = set(FEATURES.get(key, [])) | set(_LOG_EXTRA_KEYS)
    limpio = {}
    for k, v in payload.items():
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
        # compare_digest: comparacion en tiempo constante (sin canal lateral por
        # tiempo, a diferencia de `!=`, que corta en el primer caracter distinto).
        enviado = (request.headers.get("X-Admin-Token") or "").encode("utf-8")
        if not hmac.compare_digest(enviado, ADMIN_TOKEN.encode("utf-8")):
            return jsonify({"error": "no autorizado"}), 401
        return fn(*args, **kwargs)
    return wrapper


# >>> SHAP START
def _load_background_for_shap(disease: str, feats: List[str], n: int = 200) -> np.ndarray:
    """
    Carga datos reales o sintéticos para usar como background en SHAP.
    NO afecta entrenamiento ni predicción.
    """
    curated_path = os.path.join("data_curated", disease, f"{disease}_train.csv")
    processed_path = os.path.join("data_processed", f"{disease}_dataset.csv")
    path = curated_path if os.path.exists(curated_path) else processed_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"No background data for SHAP ({disease})")

    df = pd.read_csv(path, low_memory=False)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    for c in feats:
        if c not in df.columns:
            df[c] = 0

    df = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0)

    if len(df) > n:
        df = df.sample(n, random_state=42)

    return df.values.astype(float)


def _is_linear_clf(clf) -> bool:
    """Heurística: ¿el clasificador es lineal (LogReg) o basado en árboles?"""
    from sklearn.linear_model import LogisticRegression
    return isinstance(clf, LogisticRegression)


def _build_shap_explainer(disease: str):
    """
    Construye un explainer SHAP acorde al tipo de modelo ganador (A5):
    LinearExplainer para modelos lineales (LogReg) y TreeExplainer para modelos
    basados en árboles (LightGBM/RandomForest). El explainer siempre opera sobre
    el espacio escalado, igual que el clasificador dentro del pipeline.
    Se ejecuta una sola vez al iniciar el backend.
    """
    pipe = MODELS[disease]
    feats = FEATURES[disease]
    clf = pipe.named_steps["clf"]

    if _is_linear_clf(clf):
        # Lineal (LogReg): LinearExplainer necesita background en el espacio escalado.
        scaler = pipe.named_steps["scaler"]
        Xb = _load_background_for_shap(_data_disease(disease), feats, n=200)
        EXPLAINERS[disease] = shap.LinearExplainer(
            clf, scaler.transform(Xb), feature_perturbation="interventional"
        )
    else:
        # Árboles (LGBM/RandomForest): TreeExplainer en modo tree_path_dependent
        # (sin background). Es exacto y autoconsistente; pasar background dispara
        # falsos fallos del "additivity check" con LightGBM. Además evita depender
        # de una carpeta data_curated propia para las variantes (p.ej. diabetes_glucosa).
        EXPLAINERS[disease] = shap.TreeExplainer(clf)


def _shap_vector_for_positive_class(disease: str, Xt: np.ndarray, n_features: int):
    """
    Devuelve un vector 1D de valores SHAP para la CLASE POSITIVA, normalizando
    las distintas formas que devuelven los explainers (Linear vs Tree, binario):
    lista por clase, array 2D (n, features) o 3D (n, features, clases).
    """
    explainer = EXPLAINERS.get(disease)
    if explainer is None:
        return None

    if isinstance(explainer, shap.TreeExplainer):
        # check_additivity desactivado: solo rankeamos por |SHAP|, no exigimos
        # que sumen exactamente al output (irrelevante para el top-5).
        sv = explainer.shap_values(Xt, check_additivity=False)
    else:
        sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        # Lista por clase (p.ej. RandomForest) -> tomar la clase positiva
        sv = sv[-1]
    sv = np.array(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) -> clase positiva
        sv = sv[..., -1]
    sv = sv.reshape(-1)
    # Salvaguarda: si por algún motivo la longitud no calza, recortar/rellenar
    if sv.shape[0] != n_features:
        sv = np.resize(sv, n_features)
    return sv
# >>> SHAP END


# ==========================================
# CARGA DE MODELOS
# ==========================================
def _load_all():
    # Enfermedades servidas (FILES) + variantes de modelo extra (p.ej. glucosa).
    for dis in list(FILES.keys()) + EXTRA_MODEL_KEYS:
        paths = FILES.get(dis) or _model_paths(dis)
        if os.path.exists(paths["pipeline"]):
            MODELS[dis] = load(paths["pipeline"])
        if os.path.exists(paths["features"]):
            with open(paths["features"], "r", encoding="utf-8") as f:
                FEATURES[dis] = json.load(f)
        if os.path.exists(paths["metrics"]):
            try:
                with open(paths["metrics"], "r", encoding="utf-8") as f:
                    MODEL_NAMES[dis] = json.load(f).get("best_model")
            except Exception:
                pass
        # Calibrador isotonico opcional (mapea predict_proba -> prob honesta).
        cal_path = os.path.join(BASE_MODELS, f"{dis}_calibrator.pkl")
        if os.path.exists(cal_path):
            try:
                CALIBRATORS[dis] = load(cal_path)
            except Exception as e:
                print(f"⚠️ Calibrador no cargado para {dis}: {e}")
    # >>> SHAP START
    for dis in MODELS:
        try:
            _build_shap_explainer(dis)
        except Exception as e:
            print(f"⚠️ SHAP no disponible para {dis}: {e}")
    # >>> SHAP END


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


# ==========================================
# CAPA DE INTERPRETACIÓN CLÍNICA (A4)
# ==========================================
# Umbrales diagnósticos de referencia, expuestos como una capa SEPARADA y
# etiquetada que se muestra JUNTO a la probabilidad del modelo, NO encima de
# ella. La probabilidad reportada es la salida limpia del modelo de ML; estos
# indicadores no la modifican (a diferencia del antiguo `max()` con números
# mágicos). Fuentes:
#   - Glucosa / HbA1c: American Diabetes Association (ADA), Standards of Care.
#   - Presión arterial sistólica: ACC/AHA 2017 Hypertension Guideline.
def compute_clinical_flags(glucose_mgdl: float, hba1c: float, systolic: float) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []

    # --- Glucosa plasmática (ADA) ---
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

    # --- Presión arterial sistólica (ACC/AHA 2017) ---
    if systolic >= 180:
        flags.append({"indicator": "blood_pressure", "value": systolic, "category": "crisis_hipertensiva",
                      "source": "ACC/AHA", "detail": "Sistólica ≥180 mmHg: crisis hipertensiva."})
    elif systolic >= 140:
        flags.append({"indicator": "blood_pressure", "value": systolic, "category": "hipertension_grado_2",
                      "source": "ACC/AHA", "detail": "Sistólica ≥140 mmHg: hipertensión grado 2."})
    elif systolic >= 130:
        flags.append({"indicator": "blood_pressure", "value": systolic, "category": "hipertension_grado_1",
                      "source": "ACC/AHA", "detail": "Sistólica 130–139 mmHg: hipertensión grado 1."})
    elif systolic >= 120:
        flags.append({"indicator": "blood_pressure", "value": systolic, "category": "presion_elevada",
                      "source": "ACC/AHA", "detail": "Sistólica 120–129 mmHg: presión elevada."})

    return flags


# ==========================================
# FUNCIONES AUXILIARES PARA DATOS SINTÉTICOS / REALES
# ==========================================
def _sample_row_from_csv(csv_path, source_type):
    """Lee un CSV, descarta 'target', muestrea 1 fila y normaliza tipos numpy.
    Marca _source_type. Mismo formato para datos reales y sintéticos (clave para
    que el juego 'real vs sintético' presente ambas fichas idénticas en forma)."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "target" in df.columns:
            df = df.drop(columns=["target"])
        sample = df.sample(1).iloc[0].to_dict()
        sample = {k: _json_safe(v) for k, v in sample.items()}
        sample['_source_type'] = source_type
        return sample
    except Exception as e:
        print(f"Error leyendo CSV ({source_type}): {e}")
        return None


def get_real_sample(disease):
    """Una fila REAL aleatoria del split de entrenamiento curado
    (data_curated/<disease>/<disease>_train.csv), con fallback al procesado."""
    disease = disease.lower()
    csv_path = os.path.join("data_curated", disease, f"{disease}_train.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data_processed", f"{disease}_dataset.csv")
    return _sample_row_from_csv(csv_path, "real")


def get_random_sample(disease):
    """
    Busca datos SINTÉTICOS priorizando CTGAN (que son los médicamente correctos).
    """
    disease = disease.lower()
    base_dir = os.path.join("data_curated", disease)
    
    # 1. Intentar buscar específicamente CTGAN primero (Recomendado)
    ctgan_pattern = os.path.join(base_dir, f"{disease}_synthetic_ctgan*.csv")
    found_files = glob.glob(ctgan_pattern)
    
    # 2. Si no hay CTGAN, buscar cualquier otro sintético (Fallback, por si acaso)
    if not found_files:
        print(f"⚠️ No se encontró CTGAN para {disease}, buscando otros...")
        any_pattern = os.path.join(base_dir, f"{disease}_synthetic*.csv")
        found_files = glob.glob(any_pattern)

    csv_path = None
    source_type = "real"

    if found_files:
        # Tomamos el primero (ahora seguro será CTGAN si existe)
        csv_path = found_files[0]
        source_type = "synthetic"
        # Opcional: imprimir cuál estamos usando para estar seguros
        print(f"🎲 Usando datos sintéticos: {os.path.basename(csv_path)}")
    else:
        # 3. Fallback final: Datos reales procesados
        csv_path = os.path.join("data_processed", f"{disease}_dataset.csv")
        print(f"⚠️ No se hallaron sintéticos para {disease}. Usando datos reales procesados.")

    # Misma normalizacion que la ficha real (incluido NaN -> None).
    return _sample_row_from_csv(csv_path, source_type)

@app.get("/health")
def health():
    """Liveness + comprobacion REAL de la BD (antes devolvia el string fijo
    'sqlite_connected', que ademas mentiria al pasar a Postgres en el deploy)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        print(f"[WARN] /health: la BD no responde: {e}")
    payload = {
        "status": "ok" if db_ok else "degraded",
        "database": engine.url.get_backend_name(),   # sqlite | postgresql | ...
        "database_ok": db_ok,
        "models_loaded": sorted(MODELS.keys()),
    }
    return jsonify(payload), (200 if db_ok else 503)

@app.get("/metrics/<disease>")
def get_metrics(disease: str):
    disease = disease.lower()
    # Sirve enfermedades de FILES y también las variantes de modelo extra
    # (p.ej. 'diabetes_glucosa', la variante híbrida con glucosa).
    if disease not in FILES and disease not in EXTRA_MODEL_KEYS:
        return jsonify({"error": "unknown disease"}), 404
    paths = FILES.get(disease) or _model_paths(disease)
    mpath = paths["metrics"]
    if not os.path.exists(mpath):
        return jsonify({"error": "metrics not found"}), 404
    with open(mpath, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return jsonify(metrics)

@app.get("/config/<disease>")
def get_config(disease: str):
    disease = disease.lower()
    if disease not in FEATURES:
        return jsonify({"error": "features not found"}), 404
    feats = FEATURES[disease]
    ranges = { "age": [18, 100], "bmi": [15, 50], "glucose": [60, 260], "blood_pressure": [60, 130] }
    gender_opts = sorted([f.split("gender_")[1] for f in feats if f.startswith("gender_")])
    smoke_opts  = sorted([f.split("smoking_history_")[1] for f in feats if f.startswith("smoking_history_")])

    # Features OPCIONALES: las que aporta una variante con más datos (p.ej. la
    # glucosa del modelo híbrido de diabetes) y que el usuario puede rellenar o no.
    optional = []
    if disease == "diabetes" and DIABETES_GLUCOSE_KEY in FEATURES:
        optional = [f for f in FEATURES[DIABETES_GLUCOSE_KEY] if f not in feats]

    return jsonify({
        "disease": disease,
        "features": feats,
        "optional_features": optional,
        "ranges": ranges,
        "categoricals": { "gender": gender_opts, "smoking_history": smoke_opts }
    })

def _normalize_glucose_alias(payload: Dict[str, Any]) -> Dict[str, Any]:
    """`glucose` y `blood_glucose_level` son la misma variable en datasets/modelos
    distintos. Si llega solo una, se refleja en la otra (plumbing, NO regla clínica)."""
    if payload:
        if "glucose" in payload and "blood_glucose_level" not in payload:
            payload["blood_glucose_level"] = payload["glucose"]
        if "blood_glucose_level" in payload and "glucose" not in payload:
            payload["glucose"] = payload["blood_glucose_level"]
    return payload


def _build_row(key: str, payload: Dict[str, Any]):
    """Arma el vector de features en el orden del modelo (clave `key`, que puede ser
    una enfermedad o una variante como 'diabetes_glucosa'), rellena ausentes con 0 y
    captura los valores clínicos (glucosa/HbA1c/sistólica) para la capa ADA/ACC-AHA.
    Devuelve (X 2D, lista de features ausentes reportables, dict clínico)."""
    feats = FEATURES[key]
    row, missing = [], []
    clin = {"glucose": 0.0, "hba1c": 0.0, "bp": 0.0}
    for f in feats:
        val = _safe_get(payload, f)
        # Ausente = no viene o viene vacio (un input borrado en el form manda "").
        if val is None or (isinstance(val, str) and not val.strip()):
            # Las dummies (one-hot) no se reportan como ausentes; el resto si.
            # OJO: antes esto llevaba un `or "_" in f` que silenciaba features
            # reales como blood_pressure, ap_hi o blood_glucose_level.
            if not f.startswith(("gender_", "smoking_history_", "cholesterol_",
                                 "glucose_", "bp_", "ethnicity_", "race_")):
                missing.append(f)
            num = 0.0
        else:
            num = _to_float(f, val)  # no numerico -> InvalidPayload -> 400
        if f in ("glucose", "blood_glucose_level"): clin["glucose"] = num
        if f == "hba1c_level": clin["hba1c"] = num
        # Sistólica: hipertensión usa 'blood_pressure'; cardiovascular usa 'ap_hi'.
        if f in ("blood_pressure", "ap_hi"): clin["bp"] = num
        row.append(num)
    return np.array([row], dtype=float), missing, clin


def _predict_proba(key: str, X: np.ndarray):
    """Probabilidad de clase positiva para el modelo `key`. Devuelve (raw, calibrada).
    La calibrada aplica la isotónica persistida (si existe); es un mapeo monótono, así
    que preserva el orden (AUC) y la monotonía clínica. SHAP explica SIEMPRE el modelo
    crudo (la calibración es una transformación posterior del score)."""
    model = MODELS[key]
    if hasattr(model, "predict_proba"):
        raw = float(model.predict_proba(X)[0, 1])
    else:
        raw = float(model.predict(X)[0])
    raw = min(max(raw, 0.0), 1.0)
    cal_obj = CALIBRATORS.get(key)
    if cal_obj is not None:
        cal = float(cal_obj.predict([raw])[0])
        cal = min(max(cal, 0.0), 1.0)
    else:
        cal = raw
    return raw, cal


def _resolve_model_key(disease: str, payload: Dict[str, Any]) -> str:
    """Ruteo del modelo híbrido de diabetes: si el usuario aporta una glucosa válida
    y existe la variante con glucosa, se sirve esa; si no, el modelo self-report."""
    if disease == "diabetes" and DIABETES_GLUCOSE_KEY in MODELS:
        val = _safe_get(payload, "blood_glucose_level")
        try:
            if val is not None and float(val) > 0:
                return DIABETES_GLUCOSE_KEY
        except (TypeError, ValueError):
            pass
    return disease


@app.post("/predict/<disease>")
def predict(disease: str):
    disease = disease.lower()
    # Enfermedad inexistente -> 404 (coherente con /config y /metrics); el 500
    # queda solo para el caso real de servidor mal cargado.
    if disease not in FILES:
        return jsonify({"error": "unknown disease"}), 404
    if disease not in MODELS or disease not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 500

    try:
        payload = request.get_json(force=True) or {}
    except RequestEntityTooLarge:
        raise  # cuerpo por encima de MAX_CONTENT_LENGTH -> 413 (AUD-9)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400
    if not isinstance(payload, dict):
        return jsonify({"error": "el cuerpo debe ser un objeto JSON"}), 400

    payload = _normalize_glucose_alias(payload)
    # Ruteo híbrido: modelo con glucosa si el usuario la aportó, si no self-report.
    model_key = _resolve_model_key(disease, payload)
    feats = FEATURES[model_key]
    try:
        X, missing, clin = _build_row(model_key, payload)
    except InvalidPayload as e:
        return jsonify({"error": str(e)}), 400
    clinical_glucose, clinical_hba1c, clinical_bp = clin["glucose"], clin["hba1c"], clin["bp"]
    # La glucosa puede venir en el payload aunque el modelo base no la use: para la
    # capa clínica ADA se toma directamente del payload si el modelo no la capturó.
    if not clinical_glucose:
        _g = _safe_get(payload, "blood_glucose_level")
        try:
            clinical_glucose = float(_g) if _g is not None else 0.0
        except (TypeError, ValueError):
            clinical_glucose = 0.0

    model = MODELS[model_key]
    scaler = model.named_steps["scaler"]

    # 1. Predicción del modelo (cruda) + calibración isotónica.
    raw_prob, prob = _predict_proba(model_key, X)

    # >>> SHAP START
    top_features = []

    if EXPLAINERS.get(model_key) is not None:
        Xt = scaler.transform(X)
        shap_vals = _shap_vector_for_positive_class(model_key, Xt, len(feats))
        x_row = X.reshape(-1)

        # ============================
        # FILTRO: omitir SOLO género
        # ============================
        allowed_idxs = [
            i for i, name in enumerate(feats)
            if not str(name).startswith("gender_")
        ]

        # Fallback de seguridad (por si el modelo no tiene feats o algo raro)
        if not allowed_idxs:
            allowed_idxs = list(range(len(feats)))

        # Top 5 por impacto absoluto (solo dentro de allowed_idxs)
        sorted_allowed = sorted(
            allowed_idxs,
            key=lambda i: abs(shap_vals[i]),
            reverse=True
        )[:5]

        for i in sorted_allowed:
            top_features.append({
                "feature": feats[i],
                "value": float(x_row[i]),
                "shap": float(shap_vals[i]),
                "abs_shap": float(abs(shap_vals[i]))
            })
    # >>> SHAP END

    # =========================================================================
    # CAPA DE INTERPRETACIÓN CLÍNICA (A4) — DESACOPLADA DEL MODELO
    # `probability` es la salida del modelo CALIBRADA (isotónica). Los umbrales
    # diagnósticos ADA/ACC-AHA se calculan APARTE y se devuelven como
    # `clinical_flags` para mostrarse junto al número del modelo, sin alterarlo.
    # =========================================================================
    prob = min(max(prob, 0.0), 1.0)
    pred_class = 1 if prob >= 0.5 else 0

    clinical_flags = compute_clinical_flags(clinical_glucose, clinical_hba1c, clinical_bp)
    clinical_note = " ".join(f["detail"] for f in clinical_flags) or \
        "Sin indicadores clínicos por encima de umbrales de referencia."

    used_glucose = model_key == DIABETES_GLUCOSE_KEY
    log_prediction_to_db(
        disease, _loggable_payload(model_key, payload), pred_class, prob,
        model_name=MODEL_NAMES.get(model_key),
        clinical_note=clinical_note,
        top_features=top_features,
        session_id=_clean_session_id(request.headers.get("X-Session-Id")),
    )

    return jsonify({
        "disease": disease,
        "model": MODEL_NAMES.get(model_key),
        "variant": "glucosa" if used_glucose else "base",
        "used_glucose": used_glucose,
        "probability": prob,
        "raw_model_probability": raw_prob,
        "calibrated": model_key in CALIBRATORS,
        "prediction": pred_class,
        "missing_filled_as_zero": missing,
        "top_features": top_features,
        "clinical_flags": clinical_flags,
        "clinical_note": clinical_note,
        "explain_note": "La probabilidad mostrada es la salida del modelo calibrada (isotónica); `raw_model_probability` es la salida cruda. Los valores SHAP explican el modelo crudo (la calibración es un reescalado monótono posterior). Los indicadores clínicos (ADA/ACC-AHA) se muestran aparte como referencia y NO modifican la probabilidad."
    })


@app.post("/whatif/<disease>")
def whatif(disease: str):
    """Análisis contrafactual: fija un caso base y barre UNA feature sobre un rango,
    devolviendo la curva de riesgo (probabilidad calibrada) a lo largo de esa
    variable. NO se registra en la BD (no ensucia la analítica del admin) y no
    calcula SHAP. Alimenta el panel 'what-if' del simulador."""
    disease = disease.lower()
    if disease not in MODELS or disease not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 404

    try:
        body = request.get_json(force=True) or {}
    except RequestEntityTooLarge:
        raise  # cuerpo por encima de MAX_CONTENT_LENGTH -> 413 (AUD-9)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "el cuerpo debe ser un objeto JSON"}), 400

    feature = body.get("feature")
    if not feature:
        return jsonify({"error": "missing 'feature'"}), 400
    try:
        vmin, vmax = float(body["min"]), float(body["max"])
        steps = int(body.get("steps", 25))
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "invalid min/max/steps"}), 400
    if vmax <= vmin:
        return jsonify({"error": "'max' must be greater than 'min'"}), 400
    steps = max(2, min(steps, 100))

    base_raw = body.get("base") or {}
    if not isinstance(base_raw, dict):
        return jsonify({"error": "'base' debe ser un objeto JSON"}), 400
    base = dict(base_raw)
    # Mismo ruteo híbrido que /predict: si se barre la glucosa (o el base la trae),
    # se usa el modelo con glucosa; si no, el self-report.
    route_payload = _normalize_glucose_alias(dict(base))
    if feature == "blood_glucose_level":
        route_payload["blood_glucose_level"] = vmax
    model_key = _resolve_model_key(disease, route_payload)

    # Barrer una feature que el modelo no usa daba una curva plana sin sentido.
    # Se aceptan los alias de glucosa porque _normalize_glucose_alias los refleja.
    if feature not in FEATURES[model_key] + ["glucose", "blood_glucose_level"]:
        return jsonify({"error": f"'{feature}' no es una feature de {disease}"}), 400

    curve = []
    try:
        for i in range(steps):
            v = vmin + (vmax - vmin) * i / (steps - 1)
            payload = dict(base)
            payload[feature] = v
            payload = _normalize_glucose_alias(payload)
            X, _, _ = _build_row(model_key, payload)
            raw, cal = _predict_proba(model_key, X)
            curve.append({"value": round(v, 2), "probability": cal, "raw_probability": raw})
    except InvalidPayload as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "disease": disease,
        "feature": feature,
        "model": MODEL_NAMES.get(model_key),
        "variant": "glucosa" if model_key == DIABETES_GLUCOSE_KEY else "base",
        "calibrated": model_key in CALIBRATORS,
        "curve": curve,
    })

@app.get("/synthetic/<disease>")
def get_synthetic(disease):
    disease = disease.lower()
    
    # Validar que la enfermedad existe en tu sistema
    if disease not in FILES:
        return jsonify({"error": "disease not supported"}), 404
        
    sample = get_random_sample(disease)
    
    if not sample:
        # Fallback: Si no hay CSV, devolvemos un error controlado
        return jsonify({"error": "could not generate synthetic data"}), 500
        
    return jsonify(sample)


@app.get("/sample/<disease>")
def get_sample(disease):
    """Una ficha de paciente del origen pedido: ?source=real|synthetic
    (default synthetic). Alimenta el juego 'real vs sintético'."""
    disease = disease.lower()
    if disease not in FILES:
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
    if disease not in FILES:
        return jsonify({"error": "disease not supported"}), 404

    feature = request.args.get("feature")
    if not feature:
        return jsonify({"error": "feature required"}), 400

    try:
        nbins = max(5, min(40, int(request.args.get("bins", 18))))
    except (TypeError, ValueError):
        nbins = 18

    real_path = os.path.join("data_curated", disease, f"{disease}_train.csv")
    synth_files = glob.glob(os.path.join("data_curated", disease, f"{disease}_synthetic_ctgan*.csv")) \
        or glob.glob(os.path.join("data_curated", disease, f"{disease}_synthetic*.csv"))
    if not os.path.exists(real_path) or not synth_files:
        return jsonify({"error": "data not available"}), 500

    try:
        real = pd.read_csv(real_path)
        synth = pd.read_csv(synth_files[0])
        if feature not in real.columns or feature not in synth.columns:
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
    if disease not in FILES:
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
@app.get("/admin/verify")
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
    q = s.query(Prediction).filter(Prediction.disease.in_(list(FILES.keys())))
    dt_from, dt_to_excl = _parse_date_range()
    if dt_from is not None:
        q = q.filter(Prediction.timestamp >= dt_from)
    if dt_to_excl is not None:
        q = q.filter(Prediction.timestamp < dt_to_excl)
    return q


@app.get("/admin/stats")
@require_admin
def admin_stats():
    """Analítica de uso AGREGADA y anónima (sin datos personales). Solo cuenta las
    enfermedades servidas (FILES); ignora filas viejas de enfermedades retiradas.
    Acepta ?from=&to= (YYYY-MM-DD) para acotar el rango. La agregación se hace en
    Python (probabilidades, horas, top-features) para ser agnóstica al motor SQL."""
    supported = list(FILES.keys())
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
            "model": MODEL_NAMES.get(d),
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
@require_admin
def admin_predictions():
    """Lista las simulaciones más recientes (server-side, anónimas)."""
    try:
        limit = min(int(request.args.get("limit", 50)), 500)
    except (ValueError, TypeError):
        limit = 50
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
            "clinical_note": r.clinical_note,
            "session_id": r.session_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "input_data": _safe_json_loads(r.input_data),
            "top_features": _safe_json_loads(r.top_features),
        } for r in rows]

    return jsonify({"count": len(items), "items": items})


@app.get("/admin/export.csv")
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
    writer.writerow(["id", "timestamp", "disease", "model", "prediction",
                     "probability", "session_id", "clinical_note", "input_data"])
    for r in rows:
        writer.writerow([
            r.id,
            r.timestamp.isoformat() if r.timestamp else "",
            _csv_safe(r.disease),
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