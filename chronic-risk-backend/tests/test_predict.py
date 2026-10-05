# Tests de integracion de /predict (v2): los dos modos, el IMC del simplificado,
# calibracion, contrato de respuesta, SHAP y capa clinica. Usa el test_client de Flask.
import pytest

import modos as M


ENFERMEDADES = list(M.ENFERMEDADES)

CARDIO_COMPLETO = {
    "age": 62, "bmi": 31, "waist_circumference": 108, "ap_hi": 150, "ap_lo": 92,
    "total_cholesterol": 230, "hdl_cholesterol": 38, "hba1c_level": 6.1, "egfr": 72,
    "albumin_creatinine_ratio": 35, "diabetes": 0, "hypertension": 1, "high_cholesterol": 1,
    "gender_Male": 1, "gender_Female": 0, "smoking_history_never": 0,
    "smoking_history_current": 1, "smoking_history_former": 0,
}


def _predict(client, disease, payload, modo=None, **kw):
    ruta = f"/predict/{disease}" + (f"?mode={modo}" if modo else "")
    r = client.post(ruta, json=payload, **kw)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _sin(payload, *claves):
    return {k: v for k, v in payload.items() if k not in claves}


# ---------- los dos modos ----------

def test_sin_modo_sirve_el_simplificado(client, app_module, perfil_diabetes):
    d = _predict(client, "diabetes", perfil_diabetes)
    assert d["mode"] == "simplificado"
    assert d["model"] == app_module.MODEL_NAMES["diabetes_simplificado"]

def test_el_completo_sirve_su_propio_modelo(client, app_module):
    d = _predict(client, "cardiovascular", CARDIO_COMPLETO, modo="completo")
    assert d["mode"] == "completo"
    assert d["model"] == app_module.MODEL_NAMES["cardiovascular_completo"]

def test_modo_desconocido_es_400(client, perfil_diabetes):
    r = client.post("/predict/diabetes?mode=experto", json=perfil_diabetes)
    assert r.status_code == 400 and "modo" in r.get_json()["error"]

def test_enfermedad_desconocida(client):
    r = client.post("/predict/obesidad", json={"age": 40})
    assert r.status_code == 404  # AUD-3: coherente con /config y /metrics


# ---------- el IMC del simplificado sale del peso y la talla ----------

def test_el_imc_sale_del_peso_y_la_talla(client, perfil_diabetes):
    d = _predict(client, "diabetes", perfil_diabetes)
    assert d["bmi"] == pytest.approx(92 / 1.72 ** 2, abs=0.01)

def test_el_imc_dado_a_mano_vale_lo_mismo(client, perfil_diabetes):
    """Un cliente de la API que ya tiene el IMC puede mandarlo en vez del peso y la talla."""
    directo = {**_sin(perfil_diabetes, "weight", "height"), "bmi": round(92 / 1.72 ** 2, 2)}
    assert _predict(client, "diabetes", directo)["probability"] == pytest.approx(
        _predict(client, "diabetes", perfil_diabetes)["probability"], abs=1e-12)

@pytest.mark.parametrize("disease", ENFERMEDADES)
def test_mas_peso_a_igual_talla_no_baja_el_riesgo(client, disease, perfil_diabetes):
    """En la v1, a igual IMC y cintura, mas peso bajaba el riesgo de hipertension (era
    mas talla). Ahora el peso solo cuenta a traves del IMC, que no puede bajarlo."""
    base = {**perfil_diabetes, "diabetes": 0}
    probs = [_predict(client, disease, {**base, "weight": w})["probability"]
             for w in (55, 70, 85, 100, 115, 130)]
    assert probs == sorted(probs)
    assert probs[-1] > probs[0]

def test_peso_y_talla_imposibles_es_400(client, perfil_diabetes):
    r = client.post("/predict/diabetes", json={**perfil_diabetes, "weight": 250, "height": 120})
    assert r.status_code == 400 and "IMC" in r.get_json()["error"]


# ---------- datos que faltan ----------

def test_un_dato_que_falta_es_400_no_un_cero(client, perfil_diabetes):
    """La v1 rellenaba con 0 lo que faltaba (un IMC de 0 no es un paciente) y lo avisaba
    al lado del resultado. Ahora no hay resultado sin los datos del modo."""
    for falta in ("height", "age", "hypertension"):
        r = client.post("/predict/diabetes", json=_sin(perfil_diabetes, falta))
        assert r.status_code == 400 and "faltan datos" in r.get_json()["error"], falta

def test_el_tabaco_exige_una_sola_categoria(client, perfil_diabetes):
    ninguna = {**perfil_diabetes, "smoking_history_never": 0}
    dos = {**perfil_diabetes, "smoking_history_current": 1}
    for payload in (ninguna, dos):
        r = client.post("/predict/diabetes", json=payload)
        assert r.status_code == 400 and "tabaquismo" in r.get_json()["error"]

def test_lo_que_el_modelo_no_usa_puede_faltar(client, app_module):
    """Las variables sin efecto toman la mediana del train si faltan: no cambian nada."""
    sin_efecto = app_module.SIN_EFECTO["cardiovascular_completo"]
    assert sin_efecto
    completo = _predict(client, "cardiovascular", CARDIO_COMPLETO, modo="completo")
    sin_ellas = _predict(client, "cardiovascular", _sin(CARDIO_COMPLETO, *sin_efecto), modo="completo")
    assert sin_ellas["probability"] == pytest.approx(completo["probability"], abs=1e-12)


# ---------- contrato de la respuesta ----------

def test_contrato_de_respuesta(client, perfil_diabetes):
    d = _predict(client, "diabetes", perfil_diabetes)
    for key in ("disease", "mode", "model", "probability", "raw_model_probability",
                "calibrated", "risk_band", "risk_bands", "prediction", "bmi", "top_features",
                "clinical_flags", "clinical_note", "not_used", "support_warnings",
                "support_note", "explain_note"):
        assert key in d, f"falta '{key}' en la respuesta"
    assert 0.0 <= d["probability"] <= 1.0
    assert 0.0 <= d["raw_model_probability"] <= 1.0
    assert d["prediction"] in (0, 1)

def test_probabilidad_es_la_calibrada(client, app_module, perfil_diabetes):
    """`probability` debe ser exactamente la isotonica aplicada sobre la cruda."""
    d = _predict(client, "diabetes", perfil_diabetes)
    assert d["calibrated"] is True
    cal = app_module.CALIBRATORS["diabetes_simplificado"]
    assert d["probability"] == pytest.approx(float(cal.predict([d["raw_model_probability"]])[0]), abs=1e-9)


# ---------- SHAP ----------

def test_shap_omite_el_sexo_y_lo_que_no_se_usa(client, app_module):
    d = _predict(client, "cardiovascular", CARDIO_COMPLETO, modo="completo")
    tf = d["top_features"]
    assert 0 < len(tf) <= 5
    sin_efecto = set(app_module.SIN_EFECTO["cardiovascular_completo"])
    assert not any(t["feature"].startswith("gender_") or t["feature"] in sin_efecto for t in tf)
    # Ordenado por impacto absoluto descendente.
    absolutos = [t["abs_shap"] for t in tf]
    assert absolutos == sorted(absolutos, reverse=True)


# ---------- capa clinica (integrada en /predict) ----------

def _flags(d, indicador):
    return [f for f in d["clinical_flags"] if f["indicator"] == indicador]

def test_la_glucosa_genera_flag_ada_y_no_mueve_la_estimacion(client, perfil_diabetes):
    """La glucosa en ayunas define la diabetes: la interpreta la guia, no el modelo."""
    sin = _predict(client, "diabetes", perfil_diabetes)
    con = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 250})
    assert _flags(con, "glucose")[0]["category"] == "diabetes"
    assert con["probability"] == sin["probability"]

def test_presion_alta_genera_flag_acc_aha(client, perfil_diabetes):
    d = _predict(client, "hipertension", {**perfil_diabetes, "diabetes": 0, "ap_hi": 185})
    assert _flags(d, "blood_pressure")[0]["category"] == "crisis_hipertensiva"

def test_la_diastolica_tambien_cuenta(client, perfil_diabetes):
    """El objetivo de hipertension es >= 140/90: con solo la sistolica, un 132/95 salia
    como grado 1."""
    d = _predict(client, "hipertension", {**perfil_diabetes, "diabetes": 0, "ap_hi": 132, "ap_lo": 95})
    assert _flags(d, "blood_pressure")[0]["category"] == "hipertension_grado_2"

def test_valores_normales_sin_flags(client, perfil_diabetes):
    d = _predict(client, "diabetes", {**perfil_diabetes, "blood_glucose_level": 90, "hba1c_level": 5.2})
    assert d["clinical_flags"] == []
    assert d["clinical_note"].startswith("Sin indicadores")

def test_lo_que_el_modelo_no_usa_se_avisa(client, app_module):
    d = _predict(client, "cardiovascular", CARDIO_COMPLETO, modo="completo")
    aviso = _flags(d, "sin_efecto")
    assert aviso and set(aviso[0]["value"]) == set(app_module.SIN_EFECTO["cardiovascular_completo"])
    assert "no cambian esta estimación" in aviso[0]["detail"]

def test_ser_exfumador_sin_efecto_se_avisa_y_no_resta(client, app_module, perfil_diabetes):
    """Sin la restriccion de signo, en diabetes ser exfumador restaba 2,3 puntos."""
    assert "smoking_history_former" in app_module.SIN_EFECTO["diabetes_simplificado"]
    exfumador = {**perfil_diabetes, "smoking_history_never": 0, "smoking_history_former": 1}
    d = _predict(client, "diabetes", exfumador)
    assert "tabaco" in _flags(d, "sin_efecto")[0]["detail"]
    assert d["probability"] == _predict(client, "diabetes", perfil_diabetes)["probability"]
