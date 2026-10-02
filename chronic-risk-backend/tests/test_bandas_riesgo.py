"""Revision 2026-10 — bandas de riesgo por enfermedad.

El simulador pintaba bajo / moderado / alto por tercios fijos (33% y 66%) en las tres
enfermedades. En diabetes, con un 13,6% de prevalencia, eso daba "Riesgo bajo" a una
mujer sana de 25 anos con glucosa de 250 (31,8%, mas del doble que la media), dejaba
en "bajo" al 80% de los diabeticos reales del test y hacia inalcanzable el "alto" para
la variante sin glucosa, que no pasa de 54,5%.

Ahora "bajo" es quedar por debajo de la media de los datos de entrenamiento y "alto",
al menos el doble, con los tercios como tope: hipertension (36%) y cardiovascular
(50%) no cambian. La banda es una LECTURA de la probabilidad: no la modifica.
"""
import pytest

SANA_25 = {
    "age": 25, "bmi": 21, "hypertension": 0, "heart_disease": 0,
    "gender_Male": 0, "gender_Female": 1,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
HOMBRE_68 = {
    "age": 68, "bmi": 36, "hypertension": 1, "heart_disease": 1,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 0, "smoking_history_current": 0, "smoking_history_former": 1,
}


def _predict(client, disease, payload):
    r = client.post(f"/predict/{disease}", json=payload)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


# ---------- los cortes ----------

def test_diabetes_usa_la_media_y_el_doble(client, app_module):
    cortes = client.get("/config/diabetes").get_json()["risk_bands"]
    prevalencia = app_module.PREVALENCE["diabetes"]
    assert prevalencia == pytest.approx(0.1356, abs=5e-4)
    assert cortes == {"low_below": prevalencia, "high_from": 2 * prevalencia,
                      "prevalence": prevalencia, "relative_to_prevalence": True}


@pytest.mark.parametrize("disease", ["hipertension", "cardiovascular"])
def test_las_enfermedades_frecuentes_conservan_los_tercios(client, app_module, disease):
    cortes = client.get(f"/config/{disease}").get_json()["risk_bands"]
    assert (cortes["low_below"], cortes["high_from"]) == (0.33, 0.66)
    assert cortes["relative_to_prevalence"] is False
    assert cortes["prevalence"] == app_module.PREVALENCE[disease] > 0.33


def test_sin_prevalencia_quedan_los_tercios(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "PREVALENCE", {})
    assert app_module.risk_bands("diabetes") == {
        "low_below": 0.33, "high_from": 0.66, "prevalence": None, "relative_to_prevalence": False}


def test_la_variante_con_glucosa_comparte_los_cortes_de_diabetes(app_module):
    assert app_module.risk_bands("diabetes_glucosa") == app_module.risk_bands("diabetes")


def test_los_limites_de_cada_banda(app_module):
    cortes = app_module.risk_bands("diabetes")
    lo, hi = cortes["low_below"], cortes["high_from"]
    assert app_module.risk_band("diabetes", lo - 1e-9) == "low"
    assert app_module.risk_band("diabetes", lo) == "mid"
    assert app_module.risk_band("diabetes", hi - 1e-9) == "mid"
    assert app_module.risk_band("diabetes", hi) == "high"


# ---------- lo que motivo el cambio ----------

def test_glucosa_de_250_ya_no_es_riesgo_bajo(client):
    d = _predict(client, "diabetes", {**SANA_25, "blood_glucose_level": 250})
    assert 0.27 < d["probability"] < 0.33   # con tercios: "bajo"
    assert d["risk_band"] == "high"


def test_diabetes_sin_glucosa_puede_llegar_a_alto(client):
    """La variante de autorreporte no pasa de 54,5%: con el corte en 66% no llegaba nunca."""
    d = _predict(client, "diabetes", HOMBRE_68)
    assert d["variant"] == "base"
    assert d["probability"] < 0.66
    assert d["risk_band"] == "high"


def test_una_persona_sana_sigue_en_bajo(client):
    assert _predict(client, "diabetes", SANA_25)["risk_band"] == "low"
    assert _predict(client, "diabetes", {**SANA_25, "blood_glucose_level": 88})["risk_band"] == "low"


# ---------- contrato ----------

@pytest.mark.parametrize("disease,payload", [
    ("diabetes", HOMBRE_68),
    ("hipertension", {"age": 55, "bmi": 30, "weight": 90, "waist_circumference": 104,
                      "diabetes": 0, "heart_disease": 0, "high_cholesterol": 1,
                      "gender_Male": 1, "gender_Female": 0, "smoking_history_never": 1,
                      "smoking_history_current": 0, "smoking_history_former": 0}),
    ("cardiovascular", {"age": 55, "bmi": 28, "ap_hi": 135, "ap_lo": 85, "cholesterol": 2,
                        "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
                        "gender_Male": 1, "gender_Female": 0}),
])
def test_predict_devuelve_la_banda_y_sus_cortes(client, app_module, disease, payload):
    d = _predict(client, disease, payload)
    assert d["risk_bands"] == client.get(f"/config/{disease}").get_json()["risk_bands"]
    assert d["risk_band"] == app_module.risk_band(disease, d["probability"])
    assert d["risk_bands"]["low_below"] < d["risk_bands"]["high_from"]


def test_la_banda_no_toca_la_probabilidad(client, app_module, monkeypatch):
    """Mismo invariante que la capa clinica y los avisos: leer el numero sin alterarlo."""
    payload = {**SANA_25, "blood_glucose_level": 250}
    con_prevalencia = _predict(client, "diabetes", payload)
    monkeypatch.setattr(app_module, "PREVALENCE", {})
    con_tercios = _predict(client, "diabetes", payload)
    assert con_tercios["risk_band"] == "low" != con_prevalencia["risk_band"]
    assert con_tercios["probability"] == con_prevalencia["probability"]
    assert con_tercios["raw_model_probability"] == con_prevalencia["raw_model_probability"]
    assert con_tercios["prediction"] == con_prevalencia["prediction"]
