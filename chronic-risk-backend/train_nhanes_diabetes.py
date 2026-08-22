# train_nhanes_diabetes.py
# Entrena el modelo HIBRIDO de diabetes sobre el dataset NHANES:
#   - variante "self-report"  -> solo features respondibles (sin labs)
#   - variante "glucosa"       -> self-report + glucosa (opcional, semi-respondible)
# Cada variante: seleccion de modelo por CV (reusa build_models con monotonia) +
# calibracion isotonica out-of-fold. Reporta AUCs y la curva glucosa->riesgo.
# Escribe los modelos vivos: 'diabetes' (self-report, por defecto) y
# 'diabetes_glucosa' (variante con glucosa opcional).
#
# AUD-15: el split sale de data_curated (el MISMO que produce curate_and_synthesize.py
# y que usa train_models.py), no de un train_test_split propio. Antes convivian dos
# splits distintos del mismo CSV: el test que reportaban las metricas no era el test
# curado, y el fondo de SHAP -que se carga de data_curated/diabetes_train.csv- podia
# contener filas del test real del modelo.
import os, json
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from joblib import dump

from train_models import build_models, load_split_or_fallback, CV_FOLDS, SEED

# El CSV canonico lo consume curate_and_synthesize.py, que produce el split de
# data_curated; aqui solo queda como fallback de load_split_or_fallback (AUD-15).
DATASET = os.path.join("data_processed", "diabetes_dataset.csv")
MODELS_DIR = "models"
CURATED = os.path.join("data_curated", "diabetes")

# Feature sets de cada variante (HbA1c se excluye a proposito: lab puro, no
# respondible). La glucosa serica es la unica lab semi-accesible que se ofrece.
SELF_REPORT = ["age", "bmi", "hypertension", "heart_disease",
               "gender_Female", "gender_Male",
               "smoking_history_never", "smoking_history_current", "smoking_history_former"]
GLUCOSA = SELF_REPORT + ["blood_glucose_level"]


def _fit_calibrator_and_curve(pipe, X_tr, y_tr, y_te, proba_te, cv):
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y_tr)
    cal_te = cal.predict(proba_te)
    fr, mr = calibration_curve(y_te, proba_te, n_bins=10, strategy="quantile")
    fc, mc = calibration_curve(y_te, cal_te, n_bins=10, strategy="quantile")
    curve = {
        "method": "isotonic", "n_bins": 10, "strategy": "quantile",
        "brier_raw": float(brier_score_loss(y_te, proba_te)),
        "brier_calibrated": float(brier_score_loss(y_te, cal_te)),
        "raw_curve": [{"mean_pred": float(a), "frac_pos": float(b)} for a, b in zip(mr, fr)],
        "calibrated_curve": [{"mean_pred": float(a), "frac_pos": float(b)} for a, b in zip(mc, fc)],
    }
    return cal, curve


def _xy(df, features):
    """Filas con TODAS las features de la variante presentes (glucosa exige no-NaN).
    Se filtra a cada lado del split por separado: imputar la glucosa a 0 le meteria
    al modelo un pico artificial en cero, que es justo lo que se evita en el GAN."""
    sub = df.dropna(subset=features + ["target"])
    return (sub[features].values.astype(float), sub["target"].astype(int).values)


def train_variant(key, train_df, test_df, features):
    """`key` es la clave de modelo final ('diabetes' = self-report servido por
    defecto, 'diabetes_glucosa' = variante con glucosa)."""
    print(f"\n=== modelo '{key}'  ({len(features)} features) ===")
    X_tr, y_tr = _xy(train_df, features)
    X_te, y_te = _xy(test_df, features)
    n = len(y_tr) + len(y_te)
    print(f"   filas usables: {n}  (train {len(y_tr)} / test {len(y_te)})  "
          f"positivos: {int(y_tr.sum() + y_te.sum())} "
          f"({(y_tr.sum() + y_te.sum()) / n * 100:.1f}%)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    leaderboard, best_name, best_pipe, best_cv = [], None, None, -1.0
    for mname, pipe in build_models("diabetes", features).items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        m, s = float(scores.mean()), float(scores.std())
        print(f"   - {mname:14s} CV AUC = {m:.4f} (+/- {s:.4f})")
        leaderboard.append({"model": mname, "cv_auc_mean": m, "cv_auc_std": s})
        if m > best_cv:
            best_cv, best_name, best_pipe = m, mname, pipe
    leaderboard.sort(key=lambda r: r["cv_auc_mean"], reverse=True)

    best_pipe.fit(X_tr, y_tr)
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    auc_te = float(roc_auc_score(y_te, proba_te))
    report_te = classification_report(y_te, (proba_te >= 0.5).astype(int),
                                      output_dict=True, zero_division=0)
    cal, calib = _fit_calibrator_and_curve(best_pipe, X_tr, y_tr, y_te, proba_te, cv)
    print(f"   => ganador: {best_name} | AUC_test={auc_te:.3f} | "
          f"Brier {calib['brier_raw']:.4f}->{calib['brier_calibrated']:.4f}")

    dump(best_pipe, os.path.join(MODELS_DIR, f"{key}_pipeline.pkl"))
    dump(cal, os.path.join(MODELS_DIR, f"{key}_calibrator.pkl"))
    with open(os.path.join(MODELS_DIR, f"{key}_features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    meta = {"dataset": "diabetes_nhanes", "variant": key, "features": features,
            "best_model": best_name, "cv_auc": best_cv, "leaderboard": leaderboard,
            "auc": auc_te, "report": report_te, "auc_test": auc_te,
            "report_test": report_te, "calibration": calib}
    with open(os.path.join(MODELS_DIR, f"{key}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return best_pipe, cal, features


def smooth_check(pipe, cal, features):
    """Curva glucosa->riesgo calibrado con perfil fijo (verifica que sea suave)."""
    base = {"age": 50, "bmi": 30, "hypertension": 0, "heart_disease": 0,
            "gender_Male": 1, "gender_Female": 0, "smoking_history_never": 1,
            "smoking_history_current": 0, "smoking_history_former": 0}
    print("\ncurva glucosa->riesgo (50a, IMC30, hombre, no fumador):")
    prev = -1
    for g in range(80, 301, 20):
        row = dict(base); row["blood_glucose_level"] = g
        X = np.array([[row.get(f, 0) for f in features]], dtype=float)
        p = float(cal.predict(pipe.predict_proba(X)[:, 1])[0])
        flag = "" if p >= prev - 1e-9 else "  <-- BAJA"
        print(f"  glu={g:3d} -> {p*100:5.1f}%{flag}")
        prev = p


def main():
    # Mismo split que sirve el laboratorio y que alimenta el fondo de SHAP (AUD-15).
    train_df, test_df = load_split_or_fallback("diabetes")
    train_variant("diabetes", train_df, test_df, SELF_REPORT)   # self-report (por defecto)
    pipe_g, cal_g, feats_g = train_variant(                     # con glucosa
        "diabetes_glucosa", train_df, test_df, GLUCOSA)
    smooth_check(pipe_g, cal_g, feats_g)


if __name__ == "__main__":
    main()
