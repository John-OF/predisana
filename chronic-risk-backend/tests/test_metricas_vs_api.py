"""Revision 2026-10 — lo que /metricas publica es lo que la app hace.

/metricas ensenaba la sensibilidad de la clasificacion del modelo (salida cruda >= 0,5,
que es lo que evalua el entrenamiento), pero el campo `prediction` de la API cortaba la
probabilidad CALIBRADA en 0,5. No es la misma regla: los modelos se entrenan con clases
balanceadas y la calibracion deshace ese balanceo. En diabetes la pagina publicaba un
80,5% de sensibilidad y `prediction` marcaba 3 positivos de 1247 (0,6%).

Ahora `prediction` es la clase del modelo, la misma regla que se publica. Y como el
simulador no ensena ese si/no sino una banda, las metricas traen tambien `bands`: como
reparte cada banda a la gente real del test.
"""
import numpy as np
import pandas as pd
import pytest

CLAVES = [("diabetes", "diabetes"), ("diabetes_glucosa", "diabetes"),
          ("hipertension", "hipertension"), ("cardiovascular", "cardiovascular")]

HOMBRE_68 = {
    "age": 68, "bmi": 36, "hypertension": 1, "heart_disease": 1,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 0, "smoking_history_current": 0, "smoking_history_former": 1,
}


def _test_real(app_module, key, datos):
    """(cruda, calibrada, y) del modelo `key` sobre el test real de su enfermedad."""
    feats = app_module.FEATURES[key]
    df = pd.read_csv(f"data_curated/{datos}/{datos}_test.csv").dropna(subset=feats + ["target"])
    cruda = app_module.MODELS[key].predict_proba(df[feats].values.astype(float))[:, 1]
    calibrada = app_module.CALIBRATORS[key].predict(cruda)
    return cruda, calibrada, df["target"].astype(int).values


# ---------- `prediction` es la clase del modelo ----------

def test_prediction_sale_de_la_salida_cruda(client):
    """Cruda 86%, calibrada 45,5%: el modelo lo clasifica como riesgo. Antes era 0."""
    d = client.post("/predict/diabetes", json=HOMBRE_68).get_json()
    assert d["raw_model_probability"] >= 0.5 > d["probability"]
    assert d["prediction"] == 1


@pytest.mark.parametrize("glucosa", [None, 85, 110, 126, 160, 250])
def test_prediction_es_cruda_mayor_o_igual_que_un_medio(client, glucosa):
    payload = dict(HOMBRE_68) if glucosa is None else {**HOMBRE_68, "blood_glucose_level": glucosa}
    d = client.post("/predict/diabetes", json=payload).get_json()
    assert d["prediction"] == int(d["raw_model_probability"] >= 0.5)


@pytest.mark.parametrize("key,datos", CLAVES)
def test_la_sensibilidad_publicada_es_la_de_prediction(client, app_module, key, datos):
    cruda, _, y = _test_real(app_module, key, datos)
    servida = (cruda >= 0.5).astype(int)          # la regla de `prediction`
    reporte = client.get(f"/metrics/{key}").get_json()["report"]
    assert servida[y == 1].mean() == pytest.approx(reporte["1"]["recall"], abs=1e-9)
    assert (servida == y).mean() == pytest.approx(reporte["accuracy"], abs=1e-9)


def test_cortar_la_calibrada_no_es_la_regla_publicada(client, app_module):
    """Lo que hacia la API: en diabetes, una sensibilidad 100 veces menor que la publicada."""
    _, calibrada, y = _test_real(app_module, "diabetes", "diabetes")
    publicada = client.get("/metrics/diabetes").get_json()["report"]["1"]["recall"]
    assert (calibrada >= 0.5)[y == 1].mean() < 0.05 < 0.7 < publicada


# ---------- las bandas publicadas son las que se sirven ----------

@pytest.mark.parametrize("key,datos", CLAVES)
def test_los_cortes_publicados_son_los_que_sirve_la_api(client, app_module, key, datos):
    publicadas = client.get(f"/metrics/{key}").get_json()["bands"]
    servidas = app_module.risk_bands(key)
    for campo in ("low_below", "high_from", "prevalence"):
        assert publicadas[campo] == pytest.approx(servidas[campo], abs=1e-12)
    assert publicadas["relative_to_prevalence"] == servidas["relative_to_prevalence"]


@pytest.mark.parametrize("key,datos", CLAVES)
def test_el_reparto_publicado_sale_del_test_real(client, app_module, key, datos):
    _, calibrada, y = _test_real(app_module, key, datos)
    banda = np.array([app_module.risk_band(key, p) for p in calibrada])
    filas = client.get(f"/metrics/{key}").get_json()["bands"]["test"]
    assert [f["band"] for f in filas] == ["low", "mid", "high"]
    for f in filas:
        dentro = banda == f["band"]
        assert f["n"] == int(dentro.sum())
        assert f["share"] == pytest.approx(dentro.mean(), abs=1e-12)
        assert f["positive_rate"] == pytest.approx(y[dentro].mean(), abs=1e-12)
        assert f["share_of_positives"] == pytest.approx(y[dentro].sum() / y.sum(), abs=1e-12)
    assert sum(f["share"] for f in filas) == pytest.approx(1.0)
    assert sum(f["share_of_positives"] for f in filas) == pytest.approx(1.0)
    # Una banda mas alta tiene mas enfermos: si no, las bandas no dirian nada.
    tasas = [f["positive_rate"] for f in filas]
    assert tasas == sorted(tasas)
