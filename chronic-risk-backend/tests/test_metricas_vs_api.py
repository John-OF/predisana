"""Revision 2026-10 — lo que /metricas publica es lo que la app hace.

/metricas ensenaba la sensibilidad de la clasificacion del modelo (salida cruda >= 0,5,
que es lo que evalua el entrenamiento), pero el campo `prediction` de la API cortaba la
probabilidad CALIBRADA en 0,5. No es la misma regla: los modelos se entrenan con clases
balanceadas y la calibracion deshace ese balanceo. En la v1 de diabetes la pagina
publicaba un 80,5% de sensibilidad y `prediction` marcaba 3 positivos de 1247 (0,6%).

Ahora `prediction` es la clase del modelo, la misma regla que se publica. Y como el
simulador no ensena ese si/no sino una banda, las metricas traen tambien `bands`: como
reparte cada banda a la gente real del test.

Desde la v2 cada modo publica sus metricas (/metrics/<enfermedad>?mode=...) sobre las
filas del test que tienen todas sus variables, y las bandas son de la enfermedad: los
dos modos cortan con la prevalencia de todo el train, la misma que sirve la API.
"""
import numpy as np
import pandas as pd
import pytest

import modos as M

CLAVES = [(d, m) for d in M.ENFERMEDADES for m in M.MODOS]

# Hipertenso de 60 anos con IMC 29,4. Lleva los cuatro diagnosticos: cada modelo lee
# los suyos y la API ignora el resto.
HOMBRE_60 = {
    "age": 60, "weight": 90, "height": 175, "hypertension": 1, "diabetes": 0,
    "high_cholesterol": 0, "heart_disease": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
HOMBRE_60_COMPLETO = {
    **{k: v for k, v in HOMBRE_60.items() if k not in ("weight", "height")},
    "bmi": 29.4, "waist_circumference": 104, "ap_hi": 138, "ap_lo": 86,
    "total_cholesterol": 200, "hdl_cholesterol": 45, "hba1c_level": 5.7, "egfr": 85,
    "albumin_creatinine_ratio": 15, "alt": 28,
}


def _metricas(client, disease, modo):
    r = client.get(f"/metrics/{disease}?mode={modo}")
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _test_real(app_module, disease, modo):
    """(cruda, calibrada, y) del modelo de ese modo sobre el test real de su
    enfermedad: las filas que tienen todas las variables del modo."""
    clave = f"{disease}_{modo}"
    te = pd.read_csv(f"data_curated/{disease}/{disease}_test.csv")
    X = pd.DataFrame({f: te[M.columna(modo, f)] for f in app_module.FEATURES[clave]})
    ok = X.notna().all(axis=1).values
    cruda = app_module.MODELS[clave].predict_proba(X.values[ok].astype(float))[:, 1]
    calibrada = app_module.CALIBRATORS[clave].predict(cruda)
    return cruda, calibrada, te["target"].values[ok].astype(int)


# ---------- `prediction` es la clase del modelo ----------

def test_prediction_sale_de_la_salida_cruda(client):
    """Cruda 58%, calibrada 26%: el modelo lo clasifica como riesgo. Antes era 0."""
    d = client.post("/predict/diabetes", json=HOMBRE_60).get_json()
    assert d["raw_model_probability"] >= 0.5 > d["probability"]
    assert d["prediction"] == 1


@pytest.mark.parametrize("edad", [25, 45, 60, 75])
@pytest.mark.parametrize("disease,modo", CLAVES)
def test_prediction_es_cruda_mayor_o_igual_que_un_medio(client, disease, modo, edad):
    base = HOMBRE_60 if modo == "simplificado" else HOMBRE_60_COMPLETO
    r = client.post(f"/predict/{disease}?mode={modo}", json={**base, "age": edad})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d["prediction"] == int(d["raw_model_probability"] >= 0.5)


@pytest.mark.parametrize("disease,modo", CLAVES)
def test_la_sensibilidad_publicada_es_la_de_prediction(client, app_module, disease, modo):
    cruda, _, y = _test_real(app_module, disease, modo)
    metricas = _metricas(client, disease, modo)
    assert metricas["n_test"] == len(y)           # el mismo test que se publico
    servida = (cruda >= 0.5).astype(int)          # la regla de `prediction`
    reporte = metricas["report"]
    assert servida[y == 1].mean() == pytest.approx(reporte["1"]["recall"], abs=1e-9)
    assert (servida == y).mean() == pytest.approx(reporte["accuracy"], abs=1e-9)


@pytest.mark.parametrize("disease", ["diabetes", "cardiovascular"])
def test_cortar_la_calibrada_no_es_la_regla_publicada(client, app_module, disease):
    """Lo que hacia la API: en las enfermedades poco frecuentes, cortar la calibrada en
    0,5 deja la sensibilidad en menos de la cuarta parte de la publicada."""
    _, calibrada, y = _test_real(app_module, disease, "simplificado")
    publicada = _metricas(client, disease, "simplificado")["report"]["1"]["recall"]
    assert (calibrada >= 0.5)[y == 1].mean() < 0.2 < 0.75 < publicada


# ---------- las bandas publicadas son las que se sirven ----------

@pytest.mark.parametrize("disease,modo", CLAVES)
def test_los_cortes_publicados_son_los_que_sirve_la_api(client, app_module, disease, modo):
    publicadas = _metricas(client, disease, modo)["bands"]
    servidas = app_module.risk_bands(disease)
    for campo in ("low_below", "high_from", "prevalence"):
        assert publicadas[campo] == pytest.approx(servidas[campo], abs=1e-12)
    assert publicadas["relative_to_prevalence"] == servidas["relative_to_prevalence"]


@pytest.mark.parametrize("disease,modo", CLAVES)
def test_el_reparto_publicado_sale_del_test_real(client, app_module, disease, modo):
    _, calibrada, y = _test_real(app_module, disease, modo)
    banda = np.array([app_module.risk_band(disease, p) for p in calibrada])
    filas = _metricas(client, disease, modo)["bands"]["test"]
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
