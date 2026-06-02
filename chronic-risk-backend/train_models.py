# train_models.py
import os, json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump

PROCESSED_DIR = "data_processed"
CURATED_DIR = "data_curated"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DATASETS = ["diabetes", "hipertension", "obesidad", "cardiovascular"]

ALL_FEATURES = [
    "age","pregnancies","glucose","blood_pressure","skin_thickness","insulin",
    "bmi","diabetes_pedigree","hba1c_level","blood_glucose_level",
    "gender_Female","gender_Male",
    "smoking_history_current","smoking_history_former","smoking_history_never",
    "smoking_history_ever","smoking_history_not current",
    "hypertension","heart_disease"
]

def get_features_for_disease(disease_name):
    """
    Selecciona las features para cada modelo.
    """
    features = ALL_FEATURES.copy()
    
    # ✅ Evitar leakage (variables que contienen el target)
    if disease_name == "hipertension":
        if "hypertension" in features:
            features.remove("hypertension")

    if disease_name == "obesidad":
        if "bmi" in features:
            features.remove("bmi")
        
    elif disease_name == "cardiovascular":
        # Aquí sí quitamos heart_disease porque es el objetivo a prevenir (futuro)
        if "heart_disease" in features: features.remove("heart_disease")
        
    elif disease_name == "diabetes":
        pass

    return features

def load_split_or_fallback(name: str):
    """Usa train/test de data_curated si existen; si no, hace split desde data_processed."""
    curated_train = os.path.join(CURATED_DIR, name, f"{name}_train.csv")
    curated_test  = os.path.join(CURATED_DIR, name, f"{name}_test.csv")
    processed_all = os.path.join(PROCESSED_DIR, f"{name}_dataset.csv")

    if os.path.exists(curated_train) and os.path.exists(curated_test):
        print(f"   (Cargando datos curados desde {CURATED_DIR})")
        train_df = pd.read_csv(curated_train, low_memory=False)
        test_df  = pd.read_csv(curated_test,  low_memory=False)
    else:
        print(f"   (Usando fallback: split directo de {processed_all})")
        df = pd.read_csv(processed_all, low_memory=False)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["target"]
        )
    return train_df, test_df

def build_pipeline(name: str) -> Pipeline:
    class_weight = "balanced"
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, n_jobs=None)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", clf),
    ])
    return pipe

def train_one(name: str):
    print(f"\n=== Entrenando {name} ===")
    train_df, test_df = load_split_or_fallback(name)
    
    # Obtener features limpias
    current_features = get_features_for_disease(name)

    # Asegurar columnas existentes
    for c in current_features + ["target"]:
        if c not in train_df.columns:
            train_df[c] = 0
        if c not in test_df.columns:
            test_df[c] = 0

    # ==========================================================
    # SANITIZACIÓN DE EMERGENCIA (Esto arregla el error 'High')
    # ==========================================================
    print("   ...Sanitizando datos (convirtiendo textos a números)...")
    for col in current_features:
        # Intenta convertir a número. Si encuentra "High", pone NaN. Luego llena NaN con 0.
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        
    # Asegurar target numérico también
    train_df["target"] = pd.to_numeric(train_df["target"], errors='coerce').fillna(0).astype(int)
    test_df["target"] = pd.to_numeric(test_df["target"], errors='coerce').fillna(0).astype(int)
    # ==========================================================

    X_train = train_df[current_features].values
    y_train = train_df["target"].values
    X_test  = test_df[current_features].values
    y_test  = test_df["target"].values

    pipe = build_pipeline(name)
    
    # ... (todo el código anterior igual hasta la parte de las métricas)

    try:
        pipe.fit(X_train, y_train)

        # 1. Métricas para TEST
        y_proba_test = pipe.predict_proba(X_test)[:,1]
        y_pred_test = (y_proba_test >= 0.5).astype(int)
        auc_test = roc_auc_score(y_test, y_proba_test)
        report_test = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

        # 2. Métricas para TRAIN (NUEVO)
        y_proba_train = pipe.predict_proba(X_train)[:,1]
        y_pred_train = (y_proba_train >= 0.5).astype(int)
        auc_train = roc_auc_score(y_train, y_proba_train)
        report_train = classification_report(y_train, y_pred_train, output_dict=True, zero_division=0)

        # 3. Guardar modelo físicamente
        model_path = os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")
        dump(pipe, model_path)

        # 4. Consolidar TODO en un solo objeto meta (CORREGIDO)
        meta = {
            "dataset": name,
            "features": current_features,
            "auc": auc_test,        # Mantenemos 'auc' por compatibilidad con el front actual
            "report": report_test,  # Mantenemos 'report' por compatibilidad con el front actual
            "auc_test": auc_test,
            "auc_train": auc_train, 
            "report_test": report_test,
            "report_train": report_train
        }

        # Guardar el JSON con todas las métricas
        with open(os.path.join(MODELS_DIR, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Guardar lista de features por separado (como lo hace tu app.py)
        with open(os.path.join(MODELS_DIR, f"{name}_features.json"), "w", encoding="utf-8") as f:
            json.dump(current_features, f)

        print(f"{name}: AUC_Test={auc_test:.3f} | AUC_Train={auc_train:.3f} | Modelo guardado.")
    
    except Exception as e:
        print(f"Error entrenando {name}: {e}")

def main():
    for name in DATASETS:
        train_one(name)

if __name__ == "__main__":
    main()