# train_models.py
# Stack multi-modelo con seleccion por validacion cruzada (A5).
# Por cada enfermedad se entrenan varios modelos, se elige el mejor por AUC en
# cross-validation y se persiste el ganador. El score de CADA modelo se guarda en
# _metrics.json como insumo para el leaderboard de /metricas.
import os, json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict,
)
from joblib import dump

from lightgbm import LGBMClassifier

PROCESSED_DIR = "data_processed"
CURATED_DIR = "data_curated"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DATASETS = ["diabetes", "hipertension", "cardiovascular"]

# Features descartadas por baja "respondibilidad" (decision de alcance 2026-06-04).
# El simulador es educativo y de autoevaluacion: prioriza inputs que una persona
# comun puede responder (N1 autorreporte: edad, sexo, peso, talla, habitos;
# N2 medicion casera/farmacia: presion, cintura) y descarta las que exigen
# laboratorio (extraccion de sangre + orden medica). Las columnas siguen en los
# CSV (utiles para el case-study/diccionario); solo se excluyen del MODELO.
# Costo medido por ablation (CV AUC, mismo protocolo que abajo):
#   - diabetes: fuera HbA1c (lab puro); se conserva glucosa, semi-accesible via
#       glucometro/farmacia. 0.974 -> 0.926. (Quitar ambos labs caia a 0.819.)
#   - hipertension: fuera los 7 labs (juntos pesaban ~9% de importancia).
#       0.955 -> 0.946, practicamente gratis. Quedan inputs 100% respondibles.
#   - cardiovascular: nada que quitar; colesterol/glucosa ya son ordinales
#       ("te dijeron que lo tienes alto"), no valores de laboratorio.
DROP_NON_RESPONDABLE = {
    "diabetes": ["hba1c_level"],
    "hipertension": ["glucose", "hba1c_level", "cholesterol_total", "hdl",
                     "ldl", "triglycerides", "insulin"],
    "cardiovascular": [],
}

# Restricciones de monotonia para LightGBM (2026-07-05).
# Motivo: el dataset de diabetes trae la glucosa CUANTIZADA a un puñado de valores
# discretos y uno de ellos (158) quedo con 0% de positivos por como se construyo la
# fuente. LightGBM, al ser de arboles, MEMORIZABA ese pozo y devolvia riesgo ~0.02
# para glucosa 157-158 mientras sus vecinos (155, 159) daban ~0.58. Forzar monotonia
# creciente elimina el pozo de raiz y hace que el modelo respete la fisiologia:
# a mas glucosa/edad/IMC/presion, el riesgo NUNCA puede bajar. Coste de AUC minimo.
# Solo se listan features cuyo signo clinico es INEQUIVOCO (todas crecientes, +1);
# las de signo ambiguo o protector (habitos, sexo) se dejan libres (0). Aplica solo
# al candidato LightGBM (RandomForest de sklearn no soporta monotone_constraints;
# LogReg ya es monotona por construccion).
MONOTONIC_INCREASING = {
    "diabetes": ["blood_glucose_level", "hba1c_level", "age", "bmi",
                 "hypertension", "heart_disease"],
    "hipertension": ["age", "bmi", "weight", "waist_circumference", "blood_pressure"],
    "cardiovascular": ["age", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc"],
}


def _monotone_vector(disease: str, features):
    """Vector de restricciones {0,1} alineado al orden de `features` para LightGBM.
    +1 = la prediccion no puede decrecer al crecer esa feature; 0 = sin restriccion.
    El StandardScaler(with_mean=False) divide por una desviacion positiva, asi que
    preserva la direccion: monotonia en la feature escalada == monotonia en la cruda."""
    inc = set(MONOTONIC_INCREASING.get(disease, []))
    return [1 if f in inc else 0 for f in features]

# Validacion cruzada para la seleccion de modelo
CV_FOLDS = 5
SEED = 42


def get_features_for_disease(name: str, df: pd.DataFrame):
    """
    Las features son las columnas del dataset propio de la enfermedad, menos el
    target y menos las descartadas por respondibilidad (DROP_NON_RESPONDABLE).
    Con los esquemas por-enfermedad (B1) ya no hay columnas de leakage que
    recortar: cada dataset trae solo features legitimas para SU target (p.ej.
    diabetes conserva hypertension/heart_disease como comorbilidades; hipertension
    conserva blood_pressure porque el target es un score de riesgo, no la tension
    medida). El unico recorte vigente es el de laboratorio (no respondible).
    """
    drop = set(DROP_NON_RESPONDABLE.get(name, []))
    return [c for c in df.columns if c != "target" and c not in drop]


def build_models(disease: str = None, features=None) -> dict:
    """Registro de modelos candidatos. Todos comparten la misma interfaz de
    Pipeline con los pasos 'scaler' + 'clf' (nombres load-bearing para el SHAP
    de app.py). El StandardScaler(with_mean=False) es inocuo para los arboles.

    Si se pasan `disease` + `features`, el LightGBM recibe el vector de monotonia
    de esa enfermedad (ver MONOTONIC_INCREASING); sin ellos queda sin restringir."""
    lgbm_kwargs = dict(
        n_estimators=400, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1)
    if disease is not None and features is not None:
        lgbm_kwargs["monotone_constraints"] = _monotone_vector(disease, features)

    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                n_jobs=-1, random_state=SEED)),
        ]),
        "lightgbm": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LGBMClassifier(**lgbm_kwargs)),
        ]),
    }


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
            df, test_size=0.2, random_state=SEED, stratify=df["target"]
        )
    return train_df, test_df


def _sanitize(df: pd.DataFrame, features) -> pd.DataFrame:
    """Convierte features a numerico (textos -> NaN -> 0) y normaliza el target."""
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    return df


def train_one(name: str):
    print(f"\n=== Entrenando {name} ===")
    train_df, test_df = load_split_or_fallback(name)

    features = get_features_for_disease(name, train_df)
    print(f"   {len(features)} features: {features}")

    train_df = _sanitize(train_df, features)
    test_df  = _sanitize(test_df, features)

    X_train = train_df[features].values
    y_train = train_df["target"].values
    X_test  = test_df[features].values
    y_test  = test_df["target"].values

    # ---- Bake-off por cross-validation (AUC) ----
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    leaderboard = []
    best_name, best_pipe, best_cv = None, None, -1.0

    for model_name, pipe in build_models(name, features).items():
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            mean_auc, std_auc = float(scores.mean()), float(scores.std())
            print(f"   - {model_name:14s} CV AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")
        except Exception as e:
            print(f"   - {model_name:14s} FALLO en CV: {type(e).__name__}: {e}")
            continue

        leaderboard.append({
            "model": model_name,
            "cv_auc_mean": mean_auc,
            "cv_auc_std": std_auc,
        })
        if mean_auc > best_cv:
            best_cv, best_name, best_pipe = mean_auc, model_name, pipe

    if best_pipe is None:
        print(f"   ! No se pudo entrenar ningun modelo para {name}.")
        return

    leaderboard.sort(key=lambda r: r["cv_auc_mean"], reverse=True)
    print(f"   => Ganador: {best_name} (CV AUC {best_cv:.4f})")

    # ---- Reentrenar el ganador sobre todo el train y evaluar en test ----
    best_pipe.fit(X_train, y_train)

    y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    auc_test = float(roc_auc_score(y_test, y_proba_test))
    report_test = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

    y_proba_train = best_pipe.predict_proba(X_train)[:, 1]
    y_pred_train = (y_proba_train >= 0.5).astype(int)
    auc_train = float(roc_auc_score(y_train, y_proba_train))
    report_train = classification_report(y_train, y_pred_train, output_dict=True, zero_division=0)

    # ---- Calibracion de probabilidades (isotonica) ----
    # El AUC solo mide el ORDEN de los scores, no que "0.30" signifique "30% de
    # los casos asi son positivos". Sobre datos con features cuantizadas (p.ej. la
    # glucosa de diabetes) el predict_proba crudo puede estar mal calibrado.
    # Ajustamos una isotonica sobre predicciones OUT-OF-FOLD del train (sin
    # leakage) y la persistimos aparte; app.py la aplica tras predict_proba. Es un
    # mapeo MONOTONO -> preserva el AUC y la monotonia clinica del fix de glucosa.
    calibration = None
    try:
        oof_proba = cross_val_predict(
            best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(oof_proba, y_train)

        cal_test = calibrator.predict(y_proba_test)
        n_bins = 10
        frac_raw, mean_raw = calibration_curve(y_test, y_proba_test, n_bins=n_bins, strategy="quantile")
        frac_cal, mean_cal = calibration_curve(y_test, cal_test,     n_bins=n_bins, strategy="quantile")
        calibration = {
            "method": "isotonic",
            "n_bins": n_bins,
            "strategy": "quantile",
            "brier_raw": float(brier_score_loss(y_test, y_proba_test)),
            "brier_calibrated": float(brier_score_loss(y_test, cal_test)),
            "raw_curve": [{"mean_pred": float(mp), "frac_pos": float(fp)}
                          for mp, fp in zip(mean_raw, frac_raw)],
            "calibrated_curve": [{"mean_pred": float(mp), "frac_pos": float(fp)}
                                 for mp, fp in zip(mean_cal, frac_cal)],
        }
        dump(calibrator, os.path.join(MODELS_DIR, f"{name}_calibrator.pkl"))
        print(f"   Calibracion isotonica: Brier {calibration['brier_raw']:.4f} -> "
              f"{calibration['brier_calibrated']:.4f}")
    except Exception as e:
        print(f"   ! Calibracion fallo para {name}: {type(e).__name__}: {e}")

    # ---- Persistir ganador + features + metricas ----
    dump(best_pipe, os.path.join(MODELS_DIR, f"{name}_pipeline.pkl"))

    meta = {
        "dataset": name,
        "features": features,
        "best_model": best_name,
        "cv_auc": best_cv,
        "leaderboard": leaderboard,
        # Claves backward-compatible que consume el frontend actual (Metricas.jsx)
        "auc": auc_test,
        "report": report_test,
        # Detalle train/test
        "auc_test": auc_test,
        "auc_train": auc_train,
        "report_test": report_test,
        "report_train": report_train,
        # Calibracion (curva de fiabilidad raw vs calibrado + Brier) para /metricas
        "calibration": calibration,
    }
    with open(os.path.join(MODELS_DIR, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(MODELS_DIR, f"{name}_features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)

    print(f"{name}: best={best_name} | AUC_Test={auc_test:.3f} | AUC_Train={auc_train:.3f} | guardado.")


def main():
    for name in DATASETS:
        train_one(name)


if __name__ == "__main__":
    main()
