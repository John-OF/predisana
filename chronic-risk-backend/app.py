import json
import os
import sys
import sqlite3
import datetime
from typing import Dict, Any, List

import random
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

EXPLAINERS: Dict[str, Any] = {}

app = Flask(__name__)
CORS(app)

BASE_MODELS = "models"
DB_NAME = "medical_history.db"  # <--- Nombre de la Base de Datos

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
    "obesidad": {
        "pipeline": os.path.join(BASE_MODELS, "obesidad_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "obesidad_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "obesidad_metrics.json"),
    },
    "cardiovascular": {
        "pipeline": os.path.join(BASE_MODELS, "cardiovascular_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "cardiovascular_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "cardiovascular_metrics.json"),
    },
}

MODELS: Dict[str, Any] = {}
FEATURES: Dict[str, List[str]] = {}

# ==========================================
# 1. FUNCIÓN DE BASE DE DATOS
# ==========================================
def init_db():
    """Crea la tabla si no existe."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            input_data TEXT,
            prediction INTEGER,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction_to_db(disease, input_data, prediction, probability):
    """Guarda el historial de uso."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Guardamos el input como texto JSON para no complicarnos con columnas
        cursor.execute('''
            INSERT INTO predictions (disease, input_data, prediction, probability)
            VALUES (?, ?, ?, ?)
        ''', (disease, json.dumps(input_data), int(prediction), float(probability)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error guardando en BD: {e}")


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


def _build_shap_explainer(disease: str):
    """
    Construye el LinearExplainer para regresión logística.
    Se ejecuta una sola vez al iniciar el backend.
    """
    pipe = MODELS[disease]
    feats = FEATURES[disease]

    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    Xb = _load_background_for_shap(disease, feats, n=200)
    Xb_t = scaler.transform(Xb)

    EXPLAINERS[disease] = shap.LinearExplainer(
        clf,
        Xb_t,
        feature_perturbation="interventional"
    )
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
# FUNCIÓN AUXILIAR PARA DATOS SINTÉTICOS
# ==========================================
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
    clinical_bmi = 0

    for f in feats:
        val = _safe_get(payload, f)
        
        # Capturamos valores clínicos
        if f in ["glucose", "blood_glucose_level"]: clinical_glucose = float(val or 0)
        if f == "hba1c_level": clinical_hba1c = float(val or 0)
        if f == "blood_pressure": clinical_bp = float(val or 0)
        if f == "bmi": clinical_bmi = float(val or 0)

        if val is None:
            if f.startswith(("gender_", "smoking_history_", "cholesterol_", "glucose_", "bp_", "ethnicity_", "race_")) or "_" in f:
                val = 0
            else:
                val = 0
                missing.append(f)
        row.append(val)
    
    # Captura clínica directa (aunque NO esté en feats)
    bmi_in = _safe_get(payload, "bmi")
    if bmi_in is not None:
        try:
            clinical_bmi = float(bmi_in)
        except:
            clinical_bmi = 0
    else:
        clinical_bmi = 0

    # ==========================================
    # OBESIDAD: regla clínica si hay BMI
    # ==========================================
    if disease == "obesidad" and clinical_bmi > 0:
        # Regla OMS/criterio clínico
        if clinical_bmi >= 30:
            prob = 0.99
            pred_class = 1
        else:
            prob = 0.01
            pred_class = 0

        log_prediction_to_db(disease, payload, pred_class, prob)

        return jsonify({
            "disease": disease,
            "method": "clinical_rule_bmi",
            "probability": prob,
            "prediction": pred_class,
            "missing_filled_as_zero": [],
            "raw_model_probability": None,
            "top_features": [],
            "explain_note": "Para obesidad, si se proporciona BMI, se aplica criterio clínico (BMI ≥ 30). El modelo ML se usa solo cuando no hay BMI."
        })
    
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
    raw_model_probability = prob  # probabilidad base del modelo (antes de reglas clínicas)

    top_features = []
    explainer = EXPLAINERS.get(disease)

    if explainer is not None:
        Xt = scaler.transform(X)
        shap_vals = explainer.shap_values(Xt)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.array(shap_vals).reshape(-1)
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
    # REGLAS CLÍNICAS PROGRESIVAS (Dynamic Expert System)
    # Ahora la probabilidad escala suavemente con la gravedad del síntoma.
    # =========================================================================
    
    # --- DIABETES ---
    if disease == "diabetes":
        # Glucosa > 200 es diabetes casi segura.
        if clinical_glucose >= 200:
             prob = max(prob, 0.96)
        # Rango Diabético (126 - 200): Escala de 0.85 a 0.95
        elif clinical_glucose >= 126:
            extra = (clinical_glucose - 126) * 0.001 # Sube un poco por cada mg/dL extra
            prob = max(prob, 0.85 + extra)
        # Prediabetes (100 - 125): Escala de 0.40 a 0.60 (Puente para que no se caiga a 0)
        elif clinical_glucose >= 100:
            extra = (clinical_glucose - 100) * 0.008 
            prob = max(prob, 0.30 + extra)
        
        # HbA1c también empuja hacia arriba
        if clinical_hba1c >= 6.5:
            extra_a1c = (clinical_hba1c - 6.5) * 0.05
            prob = max(prob, 0.85 + extra_a1c)

    # --- HIPERTENSIÓN ---
    elif disease == "hipertension":
        # Crisis Hipertensiva (>180): Casi 100%
        if clinical_bp >= 180:
            prob = max(prob, 0.98)
            
        # Hipertensión Grado 2 (140 - 180): Escala de 0.85 a 0.97
        elif clinical_bp >= 140:
            # Por cada punto arriba de 140, sumamos 0.003 (ej. 160 -> +0.06)
            extra = (clinical_bp - 140) * 0.003
            prob = max(prob, 0.85 + extra)
            
        # Hipertensión Grado 1 (130 - 139): Escala de 0.60 a 0.80
        elif clinical_bp >= 130:
            extra = (clinical_bp - 130) * 0.02 
            prob = max(prob, 0.60 + extra)
            
        # Elevada (120 - 129): Escala de 0.30 a 0.50 (Para que no se desplome a 0)
        elif clinical_bp >= 120:
            extra = (clinical_bp - 120) * 0.02
            prob = max(prob, 0.30 + extra)

    # --- OBESIDAD ---
    elif disease == "obesidad":
        # Si no hay BMI, NO aplicar reglas; solo usar probabilidad del modelo
        pass

    # Limitar siempre a máximo 1.0 (por si la suma se pasa)
    prob = min(prob, 1.0)
    # =========================================================================

    pred_class = 1 if prob >= 0.5 else 0
    
    log_prediction_to_db(disease, payload, pred_class, prob)

    return jsonify({
        "disease": disease,
        "probability": prob,
        "prediction": pred_class,
        "missing_filled_as_zero": missing,
        "raw_model_probability": raw_model_probability,
        "top_features": top_features,
        "explain_note": "Las variables mostradas corresponden a los valores SHAP que explican la probabilidad base del modelo; la probabilidad final puede incluir reglas clínicas."
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

# Inicialización global (se ejecuta siempre)
_load_all()
init_db()

# Inicialización local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)