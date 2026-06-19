import json
import os
import sys
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
from flask_cors import CORS
from joblib import load

from sqlalchemy import (
    create_engine, inspect, text, func,
    Column, Integer, String, Float, Text as SAText, DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

EXPLAINERS: Dict[str, Any] = {}

app = Flask(__name__)
CORS(app)

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
        if request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
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

    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    Xb = _load_background_for_shap(disease, feats, n=200)
    Xb_t = scaler.transform(Xb)

    if _is_linear_clf(clf):
        EXPLAINERS[disease] = shap.LinearExplainer(
            clf, Xb_t, feature_perturbation="interventional"
        )
    else:
        # Árboles (LGBM/RandomForest): TreeExplainer en modo tree_path_dependent
        # (sin background). Es exacto y autoconsistente; pasar background dispara
        # falsos fallos del "additivity check" con LightGBM.
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
    for dis, paths in FILES.items():
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
        for key, val in sample.items():
            if isinstance(val, (np.integer, np.int64)):
                sample[key] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                sample[key] = round(float(val), 2)
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

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        if "target" in df.columns:
            df = df.drop(columns=["target"])
            
        sample = df.sample(1).iloc[0].to_dict()
        
        for key, val in sample.items():
            if isinstance(val, (np.integer, np.int64)):
                sample[key] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                sample[key] = round(float(val), 2)
        
        sample['_source_type'] = source_type
        return sample
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")
        return None

@app.get("/health")
def health():
    return jsonify({"status": "ok", "database": "sqlite_connected"})

@app.get("/metrics/<disease>")
def get_metrics(disease: str):
    disease = disease.lower()
    if disease not in FILES:
        return jsonify({"error": "unknown disease"}), 404
    mpath = FILES[disease]["metrics"]
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
    return jsonify({
        "disease": disease,
        "features": feats,
        "ranges": ranges,
        "categoricals": { "gender": gender_opts, "smoking_history": smoke_opts }
    })

@app.post("/predict/<disease>")
def predict(disease: str):
    disease = disease.lower()
    if disease not in MODELS or disease not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 500

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400
    
    # ===============================
    # Normalización de glucosa
    # ===============================
    # Para compatibilidad entre datasets:
    # - glucose
    # - blood_glucose_level
    # Si solo llega uno, se copia al otro

    if payload is not None:
        if "glucose" in payload and "blood_glucose_level" not in payload:
            payload["blood_glucose_level"] = payload["glucose"]

        if "blood_glucose_level" in payload and "glucose" not in payload:
            payload["glucose"] = payload["blood_glucose_level"]

    feats = FEATURES[disease]
    row = []
    missing = []
    
    # Variables clave para las reglas
    clinical_glucose = 0
    clinical_hba1c = 0
    clinical_bp = 0

    for f in feats:
        val = _safe_get(payload, f)

        # Capturamos valores clínicos (para la capa de interpretación ADA/ACC-AHA,
        # que se reporta APARTE y NO modifica la probabilidad del modelo).
        if f in ["glucose", "blood_glucose_level"]: clinical_glucose = float(val or 0)
        if f == "hba1c_level": clinical_hba1c = float(val or 0)
        # Sistólica: hipertensión usa 'blood_pressure'; cardiovascular usa 'ap_hi'.
        if f in ["blood_pressure", "ap_hi"]: clinical_bp = float(val or 0)

        if val is None:
            if f.startswith(("gender_", "smoking_history_", "cholesterol_", "glucose_", "bp_", "ethnicity_", "race_")) or "_" in f:
                val = 0
            else:
                val = 0
                missing.append(f)
        row.append(val)
    
    X = np.array([row], dtype=float)
    model = MODELS[disease]
    scaler = model.named_steps["scaler"]
    
    # 1. Predicción Base de la IA
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        pred = int(model.predict(X)[0])
        prob = float(pred)

    # >>> SHAP START
    top_features = []

    if EXPLAINERS.get(disease) is not None:
        Xt = scaler.transform(X)
        shap_vals = _shap_vector_for_positive_class(disease, Xt, len(feats))
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
    # La probabilidad reportada es la salida limpia del modelo de ML. Los
    # umbrales diagnósticos ADA/ACC-AHA se calculan APARTE y se devuelven como
    # `clinical_flags` para mostrarse junto al número del modelo, sin alterarlo.
    # =========================================================================
    prob = min(max(prob, 0.0), 1.0)
    pred_class = 1 if prob >= 0.5 else 0

    clinical_flags = compute_clinical_flags(clinical_glucose, clinical_hba1c, clinical_bp)
    clinical_note = " ".join(f["detail"] for f in clinical_flags) or \
        "Sin indicadores clínicos por encima de umbrales de referencia."

    log_prediction_to_db(
        disease, payload, pred_class, prob,
        model_name=MODEL_NAMES.get(disease),
        clinical_note=clinical_note,
        top_features=top_features,
        session_id=request.headers.get("X-Session-Id"),
    )

    return jsonify({
        "disease": disease,
        "model": MODEL_NAMES.get(disease),
        "probability": prob,
        "prediction": pred_class,
        "missing_filled_as_zero": missing,
        "top_features": top_features,
        "clinical_flags": clinical_flags,
        "clinical_note": clinical_note,
        "explain_note": "La probabilidad es la salida directa del modelo de ML y los valores SHAP la explican. Los indicadores clínicos (ADA/ACC-AHA) se muestran aparte como referencia y NO modifican la probabilidad."
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


# Variables continuas por enfermedad para el heatmap de correlaciones.
CORR_FEATURES = {
    "diabetes": ["age", "bmi", "blood_glucose_level", "hba1c_level"],
    "hipertension": ["age", "bmi", "weight", "waist_circumference", "blood_pressure", "glucose"],
    "cardiovascular": ["age", "bmi", "ap_hi", "ap_lo"],
}

_QUALITY_CACHE = {}


def _compute_quality(disease):
    """Calidad del sintético vs real: score SDMetrics (submuestreado, rápido) +
    matrices de correlación (pandas) para el heatmap comparado."""
    real_path = os.path.join("data_curated", disease, f"{disease}_train.csv")
    synth_files = glob.glob(os.path.join("data_curated", disease, f"{disease}_synthetic_ctgan*.csv")) \
        or glob.glob(os.path.join("data_curated", disease, f"{disease}_synthetic*.csv"))
    if not os.path.exists(real_path) or not synth_files:
        return None

    real = pd.read_csv(real_path)
    synth = pd.read_csv(synth_files[0])
    for d in (real, synth):
        if "target" in d.columns:
            d.drop(columns=["target"], inplace=True)
    cols = [c for c in real.columns if c in synth.columns]
    real, synth = real[cols], synth[cols]
    n = min(2000, len(real), len(synth))
    real_s = real.sample(n, random_state=42)
    synth_s = synth.sample(n, random_state=42)

    result = {"n_used": int(n)}

    # --- Score de fidelidad (SDMetrics) ---
    try:
        from sdmetrics.reports.single_table import QualityReport
        meta = {"columns": {c: {"sdtype": "categorical" if real_s[c].nunique() <= 10 else "numerical"} for c in cols}}
        rep = QualityReport()
        rep.generate(real_s, synth_s, meta, verbose=False)
        prop_scores = dict(zip(rep.get_properties()["Property"], rep.get_properties()["Score"]))
        result["overall"] = round(float(rep.get_score()), 4)
        result["column_shapes"] = round(float(prop_scores.get("Column Shapes", 0)), 4)
        result["column_pair_trends"] = round(float(prop_scores.get("Column Pair Trends", 0)), 4)
        try:
            details = rep.get_details("Column Shapes")
            per = [{"column": str(r["Column"]), "score": round(float(r["Score"]), 3)}
                   for _, r in details.iterrows() if pd.notna(r.get("Score"))]
            per.sort(key=lambda x: x["score"], reverse=True)
            result["per_column"] = per
        except Exception as e:
            print(f"quality details fail ({disease}): {e}")
            result["per_column"] = []
    except Exception as e:
        print(f"sdmetrics fail ({disease}): {e}")
        result["overall"] = None
        result["per_column"] = []

    # --- Correlaciones (pandas, robusto) ---
    corr_cols = [c for c in CORR_FEATURES.get(disease, []) if c in cols]
    if len(corr_cols) >= 2:
        rc = real_s[corr_cols].corr().fillna(0).round(2)
        sc = synth_s[corr_cols].corr().fillna(0).round(2)
        result["corr"] = {
            "features": corr_cols,
            "real": rc.values.tolist(),
            "synthetic": sc.values.tolist(),
        }
    return result


@app.get("/synthetic_quality/<disease>")
def get_synthetic_quality(disease):
    disease = disease.lower()
    if disease not in FILES:
        return jsonify({"error": "disease not supported"}), 404
    if disease not in _QUALITY_CACHE:
        res = _compute_quality(disease)
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


@app.get("/admin/stats")
@require_admin
def admin_stats():
    """Analítica de uso AGREGADA y anónima (sin datos personales). Solo cuenta las
    enfermedades que hoy se sirven (FILES); ignora filas viejas de enfermedades
    retiradas como `obesidad`."""
    supported = list(FILES.keys())
    with SessionLocal() as s:
        total = (
            s.query(func.count(Prediction.id))
             .filter(Prediction.disease.in_(supported)).scalar() or 0
        )
        distinct_sessions = (
            s.query(func.count(func.distinct(Prediction.session_id)))
             .filter(Prediction.session_id.isnot(None),
                     Prediction.disease.in_(supported)).scalar() or 0
        )

        agg = (
            s.query(
                Prediction.disease,
                func.count(Prediction.id),
                func.avg(Prediction.probability),
                func.sum(Prediction.prediction),
            )
            .filter(Prediction.disease.in_(supported))
            .group_by(Prediction.disease)
            .all()
        )
        by_disease = []
        for disease, n, avg_prob, positives in agg:
            n = int(n or 0)
            positives = int(positives or 0)
            by_disease.append({
                "disease": disease,
                "count": n,
                "avg_probability": round(float(avg_prob or 0), 4),
                "positives": positives,
                "positive_rate": round(positives / n, 4) if n else 0.0,
                "model": MODEL_NAMES.get(disease),
            })

        daily = (
            s.query(func.date(Prediction.timestamp), func.count(Prediction.id))
             .filter(Prediction.disease.in_(supported))
             .group_by(func.date(Prediction.timestamp))
             .order_by(func.date(Prediction.timestamp))
             .all()
        )
        timeline = [{"day": str(day), "count": int(c)} for day, c in daily]

    return jsonify({
        "total": int(total),
        "distinct_sessions": int(distinct_sessions),
        "by_disease": by_disease,
        "timeline": timeline,
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
        # Solo enfermedades servidas hoy (oculta filas viejas de `obesidad`, etc.).
        q = (s.query(Prediction)
              .filter(Prediction.disease.in_(list(FILES.keys())))
              .order_by(Prediction.id.desc()))
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


# Inicialización global (se ejecuta siempre)
_load_all()
init_db()

# Inicialización local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)