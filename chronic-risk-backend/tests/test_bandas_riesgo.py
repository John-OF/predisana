"""Revision 2026-10 — bandas de riesgo por enfermedad.

El simulador pintaba bajo / moderado / alto por tercios fijos (33% y 66%) en las tres
enfermedades. En la v1 de diabetes, con un 13,6% de prevalencia, eso daba "Riesgo bajo"
a una mujer sana de 25 anos con glucosa de 250 (31,8%, mas del doble que la media) y
dejaba en "bajo" al 80% de los diabeticos reales del test.

"Bajo" es quedar por debajo de la media de los datos de entrenamiento y "alto", al
menos el doble, con los tercios como tope. Con los datos v2 (NHANES 2017-2023, enfermedad
total) diabetes (19%) y cardiovascular (13%) usan la media y el doble; hipertension
(44%) conserva los tercios. Los dos modos de una enfermedad comparten los cortes. La
banda es una LECTURA de la probabilidad: no la modifica.
"""
import pytest

# Perfiles del modo simplificado. Llevan los cuatro diagnosticos: cada modelo lee los
# suyos (el de la propia enfermedad no es variable) y la API ignora el resto.
SANA_25 = {
    "age": 25, "weight": 55, "height": 165, "hypertension": 0, "diabetes": 0,
    "high_cholesterol": 0, "heart_disease": 0, "gender_Male": 0, "gender_Female": 1,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
HOMBRE_68 = {
    "age": 68, "weight": 110, "height": 175, "hypertension": 1, "diabetes": 1,
    "high_cholesterol": 1, "heart_disease": 1, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 0, "smoking_history_current": 0, "smoking_history_former": 1,
}


def _predict(client, disease, payload, modo=None):
    r = client.post(f"/predict/{disease}" + (f"?mode={modo}" if modo else ""), json=payload)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


# ---------- los cortes ----------

@pytest.mark.parametrize("disease,prevalencia", [("diabetes", 0.189), ("cardiovascular", 0.128)])
def test_las_enfermedades_poco_frecuentes_usan_la_media_y_el_doble(client, app_module, disease, prevalencia):
    cortes = client.get(f"/config/{disease}").get_json()["risk_bands"]
    real = app_module.PREVALENCE[disease]
    assert real == pytest.approx(prevalencia, abs=5e-3)
    assert cortes == {"low_below": real, "high_from": 2 * real,
                      "prevalence": real, "relative_to_prevalence": True}


def test_hipertension_conserva_los_tercios(client, app_module):
    cortes = client.get("/config/hipertension").get_json()["risk_bands"]
    assert (cortes["low_below"], cortes["high_from"]) == (0.33, 0.66)
    assert cortes["relative_to_prevalence"] is False
    assert cortes["prevalence"] == app_module.PREVALENCE["hipertension"] > 0.33


def test_sin_prevalencia_quedan_los_tercios(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "PREVALENCE", {})
    assert app_module.risk_bands("diabetes") == {
        "low_below": 0.33, "high_from": 0.66, "prevalence": None, "relative_to_prevalence": False}


@pytest.mark.parametrize("disease", ["diabetes", "hipertension", "cardiovascular"])
def test_los_dos_modos_comparten_los_cortes(client, disease):
    simple = client.get(f"/config/{disease}?mode=simplificado").get_json()["risk_bands"]
    assert client.get(f"/config/{disease}?mode=completo").get_json()["risk_bands"] == simple


def test_los_limites_de_cada_banda(app_module):
    cortes = app_module.risk_bands("diabetes")
    lo, hi = cortes["low_below"], cortes["high_from"]
    assert app_module.risk_band("diabetes", lo - 1e-9) == "low"
    assert app_module.risk_band("diabetes", lo) == "mid"
    assert app_module.risk_band("diabetes", hi - 1e-9) == "mid"
    assert app_module.risk_band("diabetes", hi) == "high"


# ---------- lo que se ve en el simulador ----------

@pytest.mark.parametrize("disease", ["diabetes", "hipertension", "cardiovascular"])
def test_una_persona_sana_queda_en_bajo_y_un_perfil_cargado_en_alto(client, disease):
    assert _predict(client, disease, SANA_25)["risk_band"] == "low"
    assert _predict(client, disease, HOMBRE_68)["risk_band"] == "high"


# ---------- contrato ----------

@pytest.mark.parametrize("modo", ["simplificado", "completo"])
@pytest.mark.parametrize("disease", ["diabetes", "hipertension", "cardiovascular"])
def test_predict_devuelve_la_banda_y_sus_cortes(client, app_module, disease, modo):
    payload = HOMBRE_68 if modo == "simplificado" else {
        **HOMBRE_68, "bmi": 35.9, "waist_circumference": 118, "ap_hi": 142, "ap_lo": 88,
        "total_cholesterol": 215, "hdl_cholesterol": 40, "hba1c_level": 6.0, "egfr": 70,
        "albumin_creatinine_ratio": 25}
    d = _predict(client, disease, payload, modo)
    assert d["risk_bands"] == client.get(f"/config/{disease}").get_json()["risk_bands"]
    assert d["risk_band"] == app_module.risk_band(disease, d["probability"])
    assert d["risk_bands"]["low_below"] < d["risk_bands"]["high_from"]


def test_la_banda_no_toca_la_probabilidad(client, app_module, monkeypatch):
    """Mismo invariante que la capa clinica y los avisos: leer el numero sin alterarlo."""
    con_prevalencia = _predict(client, "diabetes", HOMBRE_68)
    monkeypatch.setattr(app_module, "PREVALENCE", {})
    con_tercios = _predict(client, "diabetes", HOMBRE_68)
    assert con_tercios["risk_bands"] != con_prevalencia["risk_bands"]
    assert con_tercios["probability"] == con_prevalencia["probability"]
    assert con_tercios["raw_model_probability"] == con_prevalencia["raw_model_probability"]
    assert con_tercios["prediction"] == con_prevalencia["prediction"]
