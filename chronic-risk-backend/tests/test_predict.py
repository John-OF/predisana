# Tests de integracion de /predict: ruteo hibrido, alias, calibracion,
# contrato de respuesta, SHAP y capa clinica. Usa el test_client de Flask.
import pytest


def _predict(client, disease, payload, **kw):
    r = client.post(f"/predict/{disease}", json=payload, **kw)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


# ---------- ruteo hibrido con/sin glucosa ----------

def test_sin_glucosa_sirve_modelo_base(client, app_module, perfil_diabetes):
    d = _predict(client, "diabetes", perfil_diabetes)
    assert d["variant"] == "base"
    assert d["used_glucose"] is False
    assert d["model"] == app_module.MODEL_NAMES.get("diabetes")

def test_con_glucosa_sirve_variante(client, app_module, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 160})
    assert d["variant"] == "glucosa"
    assert d["used_glucose"] is True
    assert d["model"] == app_module.MODEL_NAMES.get("diabetes_glucosa")

def test_alias_glucose_equivale_a_blood_glucose(client, perfil_diabetes):
    """El mismo valor por cualquiera de los dos nombres da la misma probabilidad."""
    d1 = _predict(client, "diabetes", {**perfil_diabetes, "glucose": 160})
    d2 = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 160})
    assert d1["variant"] == d2["variant"] == "glucosa"
    assert d1["probability"] == pytest.approx(d2["probability"], abs=1e-12)


# ---------- contrato de la respuesta ----------

def test_contrato_de_respuesta(client, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 120})
    for key in ("disease", "model", "variant", "used_glucose", "probability",
                "raw_model_probability", "calibrated", "prediction",
                "missing_filled_as_zero", "top_features", "clinical_flags",
                "clinical_note", "explain_note"):
        assert key in d, f"falta '{key}' en la respuesta"
    assert 0.0 <= d["probability"] <= 1.0
    assert 0.0 <= d["raw_model_probability"] <= 1.0
    assert d["prediction"] in (0, 1)

def test_probabilidad_es_la_calibrada(client, app_module, perfil_diabetes):
    """`probability` debe ser exactamente la isotonica aplicada sobre la cruda."""
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 140})
    assert d["calibrated"] is True
    cal = app_module.CALIBRATORS["diabetes_glucosa"]
    esperada = float(cal.predict([d["raw_model_probability"]])[0])
    assert d["probability"] == pytest.approx(esperada, abs=1e-9)

def test_features_ausentes_se_reportan(client, perfil_diabetes):
    payload = dict(perfil_diabetes)
    del payload["bmi"]
    d = _predict(client, "diabetes", payload)
    assert "bmi" in d["missing_filled_as_zero"]
    # Las dummies one-hot no se reportan como ausentes.
    assert not any(f.startswith(("gender_", "smoking_history_"))
                   for f in d["missing_filled_as_zero"])

def test_enfermedad_desconocida(client):
    r = client.post("/predict/obesidad", json={"age": 40})
    assert r.status_code == 404  # AUD-3: coherente con /config y /metrics


# ---------- SHAP ----------

def test_shap_omite_genero_y_limita_a_5(client, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 180})
    tf = d["top_features"]
    assert 0 < len(tf) <= 5
    assert not any(item["feature"].startswith("gender_") for item in tf)
    # Ordenado por impacto absoluto descendente.
    absolutos = [item["abs_shap"] for item in tf]
    assert absolutos == sorted(absolutos, reverse=True)


# ---------- capa clinica (integrada en /predict) ----------

def test_glucosa_alta_genera_flag_ada(client, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 250})
    glucosa_flags = [f for f in d["clinical_flags"] if f["indicator"] == "glucose"]
    assert glucosa_flags and glucosa_flags[0]["category"] == "diabetes"
    assert d["clinical_note"] != ""

def test_presion_alta_genera_flag_acc_aha(client):
    d = _predict(client, "hipertension",
                 {"age": 60, "bmi": 30, "weight": 85, "waist_circumference": 100,
                  "blood_pressure": 185, "gender_Male": 1, "gender_Female": 0})
    bp_flags = [f for f in d["clinical_flags"] if f["indicator"] == "blood_pressure"]
    assert bp_flags and bp_flags[0]["category"] == "crisis_hipertensiva"

def test_valores_normales_sin_flags(client, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 90})
    assert d["clinical_flags"] == []
    assert d["clinical_note"].startswith("Sin indicadores")


# ---------- monotonia riesgo <-> glucosa (fix 7a6f86b) ----------

def test_riesgo_no_baja_al_subir_glucosa(client, perfil_diabetes):
    """A perfil fijo, mas glucosa nunca puede dar menos riesgo (restriccion de
    monotonia en LightGBM + calibracion isotonica, que preserva el orden)."""
    probs = []
    for g in (90, 110, 130, 150, 170, 200, 240, 280):
        d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": g})
        probs.append(d["probability"])
    for antes, despues in zip(probs, probs[1:]):
        assert despues >= antes - 1e-9, f"el riesgo bajo: {probs}"
