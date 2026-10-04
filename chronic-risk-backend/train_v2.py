# train_v2.py
# Entrenamiento v2 (revision 2026-10). Por cada enfermedad y modo compiten LogReg,
# LightGBM y RandomForest, cada uno con sus hiperparametros ajustados por validacion
# cruzada ANIDADA (la busqueda se repite dentro de cada pliegue exterior, asi que el
# AUC con el que se comparan no esta inflado por haber elegido los parametros sobre
# los mismos datos). Y solo puede ganar quien pasa el FILTRO DE VALIDACION:
#   - signos clinicos: subir una variable con sentido +1 (edad, IMC, presion...) no
#     puede bajar el riesgo de nadie del train, ni subirlo una con sentido -1 (HDL,
#     eGFR). En la v1 esto se descubria despues (el peso de hipertension, el tabaco de
#     cardiovascular); ahora es un requisito para ganar.
#   - subgrupos: AUC >= 0,55 por sexo y por tramo de edad, para que no gane un
#     modelo que solo ordena bien por edad.
# El ganador se calibra con la isotonica centrada (fit_calibrator) sobre sus
# predicciones out-of-fold y se evalua UNA vez en el test, tambien con los pesos
# muestrales de NHANES (representativo de los adultos de EE. UU.).
#
# Escribe data_curated/v2/<enfermedad>/ (el reparto) y models/v2/. No toca la v1.
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import modos_v2 as M
from monotonic_logreg import MonotonicLogisticRegression
from risk_banding import band_report
from train_models import fit_calibrator

DATA = os.path.join("data_processed", "v2")
CURATED = os.path.join("data_curated", "v2")
MODELS = os.path.join("models", "v2")
SEED = 42
AUC_MIN_SUBGRUPO = 0.55
TRAMOS_EDAD = ((18, 39), (40, 59), (60, 80))


def repartir(enfermedad):
    """Un solo reparto por enfermedad (estratificado por objetivo y ciclo) que comparten
    los dos modos: cada uno se queda con las filas que tienen todas sus variables."""
    d = pd.read_csv(os.path.join(DATA, f"{enfermedad}_dataset.csv"))
    estrato = d["target"].astype(str) + "_" + d["ciclo"]
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=estrato)
    out = os.path.join(CURATED, enfermedad)
    os.makedirs(out, exist_ok=True)
    tr.to_csv(os.path.join(out, f"{enfermedad}_train.csv"), index=False)
    te.to_csv(os.path.join(out, f"{enfermedad}_test.csv"), index=False)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def matriz(d, enfermedad, modo):
    feats = M.features(enfermedad, modo)
    X = pd.DataFrame({f: d[M.columna(modo, f)] for f in feats})
    ok = X.notna().all(axis=1).values
    peso = d["peso_entrevista" if modo == "simplificado" else "peso_examen"].values
    return X.values[ok].astype(float), d["target"].values[ok], d[ok].reset_index(drop=True), peso[ok]


def candidatos(enfermedad, modo, rapido):
    sentidos = M.sentidos(enfermedad, modo)

    def pipe(clf):
        return Pipeline([("prep", M.LogColumnas(M.columnas_log(enfermedad, modo))),
                         ("scaler", StandardScaler(with_mean=False)), ("clf", clf)])

    lgbm = {"clf__n_estimators": [100, 300, 600], "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__num_leaves": [7, 15, 31], "clf__min_child_samples": [20, 50, 100],
            "clf__reg_lambda": [0.0, 1.0, 5.0]}
    return {
        "logreg": (pipe(MonotonicLogisticRegression(sentidos, class_weight="balanced", max_iter=3000)),
                   {"clf__C": [0.1, 1.0] if rapido else [0.01, 0.1, 1.0, 10.0]}, None),
        "lightgbm": (pipe(LGBMClassifier(monotone_constraints=sentidos, class_weight="balanced",
                                         random_state=SEED, n_jobs=1, verbose=-1)),
                     lgbm, 3 if rapido else 12),
        "random_forest": (pipe(RandomForestClassifier(n_estimators=100 if rapido else 200,
                                                      class_weight="balanced",
                                                      random_state=SEED, n_jobs=1)),
                          {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [5, 20, 50]},
                          2 if rapido else None),
    }


def buscador(pipe, grid, n_iter, pliegues):
    cv = StratifiedKFold(pliegues, shuffle=True, random_state=SEED)
    if n_iter:
        return RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                  random_state=SEED, n_jobs=-1)
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)


def oof_anidado(pipe, grid, n_iter, X, y, exterior, interior):
    """Predicciones out-of-fold con la busqueda de parametros DENTRO de cada pliegue."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(exterior, shuffle=True, random_state=SEED).split(X, y):
        b = buscador(pipe, grid, n_iter, interior).fit(X[tr], y[tr])
        oof[va] = b.predict_proba(X[va])[:, 1]
    return oof


def auc_subgrupos(y, p, filas):
    out = {"hombres": filas["gender_Male"].values == 1, "mujeres": filas["gender_Female"].values == 1}
    for lo, hi in TRAMOS_EDAD:
        out[f"edad_{lo}_{hi}"] = filas["age"].between(lo, hi).values
    return {k: (round(float(roc_auc_score(y[m], p[m])), 4) if len(set(y[m])) == 2 else None)
            for k, m in out.items()}


def efectos(modelo, X, enfermedad, modo):
    """Que hace mover UNA variable (binarias: de 0 a 1; continuas: + 1 DE) a cada
    persona del train. Devuelve la fraccion a la que le mueve el riesgo al reves de su
    sentido clinico, y las variables que no mueven a nadie: las que las restricciones
    dejaron en cero porque los datos no dan senal en el sentido clinico."""
    base = modelo.predict_proba(X)[:, 1]
    al_reves, sin_efecto = {}, []
    for j, (f, s) in enumerate(zip(M.features(enfermedad, modo), M.sentidos(enfermedad, modo))):
        X2 = X.copy()
        if set(np.unique(X[:, j])) <= {0.0, 1.0}:
            X0 = X.copy()
            X0[:, j], X2[:, j] = 0.0, 1.0
            cambio = modelo.predict_proba(X2)[:, 1] - modelo.predict_proba(X0)[:, 1]
        else:
            X2[:, j] = X[:, j] + X[:, j].std()
            cambio = modelo.predict_proba(X2)[:, 1] - base
        # Un grupo one-hot (sexo, tabaco) no cuenta: de dos columnas complementarias el
        # arbol usa una y la otra sale "sin efecto" aunque el sexo si pese.
        if np.abs(cambio).max() <= 1e-12 and not f.startswith(("gender_", "smoking_history_")):
            sin_efecto.append(f)
        if s != 0:
            al_reves[f] = round(float((s * cambio < -1e-9).mean()), 4)
    return al_reves, sin_efecto


def curva(y, p):
    frac, media = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return [{"mean_pred": float(a), "frac_pos": float(b)} for a, b in zip(media, frac)]


def entrenar(enfermedad, modo, tr, te, rapido):
    t0 = time.time()
    X, y, filas, _ = matriz(tr, enfermedad, modo)
    Xte, yte, filas_te, wte = matriz(te, enfermedad, modo)
    exterior, interior = (3, 2) if rapido else (5, 3)
    print(f"\n=== {enfermedad} / {modo}: train {len(y)} ({y.mean():.1%}), test {len(yte)} ===")

    tabla = []
    for nombre, (pipe, grid, n_iter) in candidatos(enfermedad, modo, rapido).items():
        oof = oof_anidado(pipe, grid, n_iter, X, y, exterior, interior)
        final = buscador(pipe, grid, n_iter, interior).fit(X, y)
        signos, sin_efecto = efectos(final.best_estimator_, X, enfermedad, modo)
        subgrupos = auc_subgrupos(y, oof, filas)
        motivos = [f"{f} al reves en el {v:.1%}" for f, v in signos.items() if v > 0]
        motivos += [f"AUC {k} {v}" for k, v in subgrupos.items() if v is not None and v < AUC_MIN_SUBGRUPO]
        fila = {"model": nombre, "cv_auc_mean": float(roc_auc_score(y, oof)),
                "best_params": {k.replace("clf__", ""): v for k, v in final.best_params_.items()},
                "violaciones_signo": signos, "sin_efecto": sin_efecto,
                "auc_subgrupos_cv": subgrupos, "pasa_filtro": not motivos, "motivos": motivos}
        tabla.append((fila, final.best_estimator_))
        estado = "pasa" if not motivos else "NO pasa: " + "; ".join(motivos)
        print(f"   - {nombre:13s} AUC anidado {fila['cv_auc_mean']:.4f} | {estado}")

    validos = [(f, m) for f, m in tabla if f["pasa_filtro"]]
    if not validos:
        raise SystemExit(f"{enfermedad}/{modo}: ningun candidato pasa el filtro de validacion")
    ganador_fila, ganador = max(validos, key=lambda t: t[0]["cv_auc_mean"])

    # Calibracion sobre el OOF del ganador con sus parametros finales.
    cv = StratifiedKFold(exterior, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(ganador), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    calibrador = fit_calibrator(oof, y)

    crudo = ganador.predict_proba(Xte)[:, 1]
    cal = calibrador.predict(crudo)
    prevalencia = float(y.mean())
    reporte = classification_report(yte, (crudo >= 0.5).astype(int), output_dict=True, zero_division=0)
    meta = {
        "dataset": f"nhanes_v2_{enfermedad}", "version": 2, "modo": modo,
        "features": M.features(enfermedad, modo),
        "definitorias": list(M.DEFINITORIAS[enfermedad]),
        "objetivo": {"diabetes": "diagnosticada, HbA1c >= 6,5% o glucosa en ayunas >= 126 mg/dL",
                     "hipertension": "diagnosticada, >= 140/90 mmHg medida o medicacion",
                     "cardiovascular": "cardiopatia coronaria, angina, infarto, insuficiencia "
                                       "cardiaca o ictus autorreportados"}[enfermedad],
        "n_train": int(len(y)), "n_test": int(len(yte)), "prevalencia": prevalencia,
        "best_model": ganador_fila["model"], "cv_auc": ganador_fila["cv_auc_mean"],
        # Variables que el modelo servido no usa (las restricciones las dejaron en cero):
        # si el usuario las marca, la capa clinica tiene que decirlo.
        "sin_efecto": ganador_fila["sin_efecto"],
        "leaderboard": sorted((f for f, _ in tabla), key=lambda f: -f["cv_auc_mean"]),
        "auc": float(roc_auc_score(yte, crudo)), "auc_test": float(roc_auc_score(yte, crudo)),
        "auc_test_ponderado": float(roc_auc_score(yte, crudo, sample_weight=wte)),
        "auc_train": float(roc_auc_score(y, ganador.predict_proba(X)[:, 1])),
        "report": reporte, "report_test": reporte,
        "subgrupos_test": auc_subgrupos(yte, crudo, filas_te),
        "calibration": {
            "method": "isotonic", "n_bins": 10, "strategy": "quantile",
            "brier_raw": float(brier_score_loss(yte, crudo)),
            "brier_calibrated": float(brier_score_loss(yte, cal)),
            "brier_calibrated_ponderado": float(brier_score_loss(yte, cal, sample_weight=wte)),
            "raw_curve": curva(yte, crudo), "calibrated_curve": curva(yte, cal),
        },
        "bands": band_report(yte, cal, prevalencia),
    }
    os.makedirs(MODELS, exist_ok=True)
    clave = f"{enfermedad}_{modo}"
    dump(ganador, os.path.join(MODELS, f"{clave}_pipeline.pkl"))
    dump(calibrador, os.path.join(MODELS, f"{clave}_calibrator.pkl"))
    with open(os.path.join(MODELS, f"{clave}_features.json"), "w", encoding="utf-8") as f:
        json.dump(meta["features"], f, ensure_ascii=False)
    with open(os.path.join(MODELS, f"{clave}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    c = meta["calibration"]
    print(f"   => {meta['best_model']} | AUC test {meta['auc_test']:.4f} (ponderado {meta['auc_test_ponderado']:.4f})"
          f" | Brier {c['brier_raw']:.4f} -> {c['brier_calibrated']:.4f}"
          f" | sin efecto: {meta['sin_efecto'] or 'ninguna'} | {time.time() - t0:.0f} s")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento v2 por enfermedad y modo")
    parser.add_argument("--only", default="", help="enfermedades separadas por comas")
    parser.add_argument("--rapido", action="store_true", help="rejillas y pliegues minimos (depurar)")
    args = parser.parse_args()
    enfermedades = [e.strip() for e in args.only.split(",") if e.strip()] or list(M.ENFERMEDADES)
    for enfermedad in enfermedades:
        tr, te = repartir(enfermedad)
        for modo in M.MODOS:
            entrenar(enfermedad, modo, tr, te, args.rapido)


if __name__ == "__main__":
    main()
