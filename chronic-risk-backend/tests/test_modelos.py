"""Modelos (train_models.py, v2): lo que el filtro de validacion exige al ganador, y
que lo publicado en *_metrics.json sea lo que da el modelo que se sirve.

En la v1 los fallos se descubrian despues de entrenar (el peso protector de
hipertension, el tabaco de cardiovascular). En la v2 son requisitos para ganar: un
candidato que mueve el riesgo al reves de su sentido clinico no puede ser el servido,
y RandomForest, que no admite restricciones, queda fuera en todos los modos.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

import modos as M

MODELOS = [(e, m) for e in M.ENFERMEDADES for m in M.MODOS]


def _artefactos(enfermedad, modo):
    base = os.path.join("models", f"{enfermedad}_{modo}")
    with open(base + "_metrics.json", encoding="utf-8") as f:
        metricas = json.load(f)
    with open(base + "_features.json", encoding="utf-8") as f:
        features = json.load(f)
    return (joblib.load(base + "_pipeline.pkl"), joblib.load(base + "_calibrator.pkl"),
            features, metricas)


def _test(enfermedad, modo):
    d = pd.read_csv(os.path.join("data_curated", enfermedad, f"{enfermedad}_test.csv"))
    X = pd.DataFrame({f: d[M.columna(modo, f)] for f in M.features(enfermedad, modo)})
    ok = X.notna().all(axis=1)
    return X[ok].values.astype(float), d.loc[ok, "target"].values


@pytest.fixture(scope="module", params=MODELOS, ids=lambda p: f"{p[0]}-{p[1]}")
def modelo(request):
    enfermedad, modo = request.param
    return (enfermedad, modo) + _artefactos(enfermedad, modo) + _test(enfermedad, modo)


# ---------- modos ----------

def test_ningun_modelo_usa_las_variables_que_definen_su_enfermedad():
    for enfermedad in M.ENFERMEDADES:
        for modo in M.MODOS:
            assert not set(M.DEFINITORIAS[enfermedad]) & set(M.features(enfermedad, modo))
        # ni la propia enfermedad como variable
        nombre = {"diabetes": "diabetes", "hipertension": "hypertension",
                  "cardiovascular": "heart_disease", "renal": "kidney_disease"}[enfermedad]
        assert all(nombre not in M.features(enfermedad, m) for m in M.MODOS)


def test_el_simplificado_es_un_subconjunto_del_completo():
    for enfermedad in M.ENFERMEDADES:
        assert set(M.features(enfermedad, "simplificado")) < set(M.features(enfermedad, "completo"))


def test_el_simplificado_lee_el_imc_autodeclarado():
    assert M.columna("simplificado", "bmi") == "bmi_autodeclarado"
    assert M.columna("completo", "bmi") == "bmi"


def test_el_log_no_cambia_el_orden():
    t = M.LogColumnas((1,)).fit(np.zeros((1, 2)))
    x = np.array([[5.0, 0.0], [5.0, 30.0], [5.0, 3000.0]])
    y = t.transform(x)
    assert (y[:, 0] == 5.0).all() and (np.diff(y[:, 1]) > 0).all()


# ---------- los modelos servidos ----------

def test_las_variables_servidas_son_las_del_modo(modelo):
    enfermedad, modo, _, _, features, metricas, _, _ = modelo
    assert features == M.features(enfermedad, modo) == metricas["features"]
    assert metricas["modo"] == modo and metricas["version"] == 2


def test_el_ganador_paso_el_filtro_y_randomforest_no(modelo):
    *_, metricas, _, _ = modelo
    filas = {f["model"]: f for f in metricas["leaderboard"]}
    assert filas[metricas["best_model"]]["pasa_filtro"]
    assert not filas["random_forest"]["pasa_filtro"]
    validos = [f["cv_auc_mean"] for f in filas.values() if f["pasa_filtro"]]
    assert metricas["cv_auc"] == max(validos)


def test_ninguna_variable_mueve_el_riesgo_al_reves_en_el_test(modelo):
    enfermedad, modo, pipe, _, _, _, X, _ = modelo
    base = pipe.predict_proba(X)[:, 1]
    for j, (f, s) in enumerate(zip(M.features(enfermedad, modo), M.sentidos(enfermedad, modo))):
        if s == 0:
            continue
        X2 = X.copy()
        X2[:, j] = X[:, j] + X[:, j].std()
        assert (s * (pipe.predict_proba(X2)[:, 1] - base) >= -1e-9).all(), f


def test_lo_publicado_es_lo_que_da_el_modelo_servido(modelo):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    *_, pipe, cal, _, metricas, X, y = modelo
    crudo = pipe.predict_proba(X)[:, 1]
    assert len(y) == metricas["n_test"]
    assert roc_auc_score(y, crudo) == pytest.approx(metricas["auc_test"], abs=1e-12)
    assert brier_score_loss(y, cal.predict(crudo)) == pytest.approx(
        metricas["calibration"]["brier_calibrated"], abs=1e-12)


def test_discrimina_y_la_calibracion_no_empeora(modelo):
    *_, metricas, _, _ = modelo
    assert metricas["auc_test"] > 0.75
    c = metricas["calibration"]
    assert c["brier_calibrated"] <= c["brier_raw"] + 1e-4


def test_el_calibrador_no_afirma_certeza_ni_tiene_mesetas(modelo):
    _, _, _, cal, *_ = modelo
    assert 0 < cal.y_thresholds_.min() and cal.y_thresholds_.max() < 1
    assert (np.diff(cal.y_thresholds_) > 0).all()
