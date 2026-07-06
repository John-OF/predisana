# Tests unitarios de los helpers puros de app.py (sin HTTP).
import pytest


# ---------- alias glucose <-> blood_glucose_level (plumbing, no regla clinica) ----------

def test_alias_glucose_se_refleja(app_module):
    p = app_module._normalize_glucose_alias({"glucose": 120})
    assert p["blood_glucose_level"] == 120

def test_alias_blood_glucose_se_refleja(app_module):
    p = app_module._normalize_glucose_alias({"blood_glucose_level": 130})
    assert p["glucose"] == 130

def test_alias_no_pisa_si_llegan_ambas(app_module):
    p = app_module._normalize_glucose_alias({"glucose": 100, "blood_glucose_level": 200})
    assert p["glucose"] == 100 and p["blood_glucose_level"] == 200

def test_alias_payload_vacio(app_module):
    assert app_module._normalize_glucose_alias({}) == {}


# ---------- ruteo del modelo hibrido de diabetes ----------

def test_ruteo_con_glucosa_valida(app_module):
    assert app_module._resolve_model_key("diabetes", {"blood_glucose_level": 150}) == "diabetes_glucosa"

def test_ruteo_sin_glucosa(app_module):
    assert app_module._resolve_model_key("diabetes", {"age": 40}) == "diabetes"

def test_ruteo_glucosa_cero_no_activa_variante(app_module):
    assert app_module._resolve_model_key("diabetes", {"blood_glucose_level": 0}) == "diabetes"

def test_ruteo_glucosa_invalida_cae_al_base(app_module):
    assert app_module._resolve_model_key("diabetes", {"blood_glucose_level": "abc"}) == "diabetes"

def test_ruteo_otras_enfermedades_ignora_glucosa(app_module):
    assert app_module._resolve_model_key("hipertension", {"blood_glucose_level": 150}) == "hipertension"


# ---------- capa clinica ADA / ACC-AHA (separada del modelo) ----------

@pytest.mark.parametrize("glucosa,categoria", [
    (250, "diabetes"),        # >=200: compatible con diabetes
    (150, "diabetes"),        # >=126: criterio de diabetes
    (110, "prediabetes"),     # 100-125
    (90, None),               # normal: sin flag
])
def test_flags_glucosa_ada(app_module, glucosa, categoria):
    flags = app_module.compute_clinical_flags(glucosa, 0, 0)
    cats = [f["category"] for f in flags if f["indicator"] == "glucose"]
    assert cats == ([categoria] if categoria else [])

@pytest.mark.parametrize("hba1c,categoria", [
    (7.0, "diabetes"),
    (6.0, "prediabetes"),
    (5.0, None),
])
def test_flags_hba1c_ada(app_module, hba1c, categoria):
    flags = app_module.compute_clinical_flags(0, hba1c, 0)
    cats = [f["category"] for f in flags if f["indicator"] == "hba1c"]
    assert cats == ([categoria] if categoria else [])

@pytest.mark.parametrize("sistolica,categoria", [
    (185, "crisis_hipertensiva"),
    (150, "hipertension_grado_2"),
    (135, "hipertension_grado_1"),
    (125, "presion_elevada"),
    (110, None),
])
def test_flags_presion_acc_aha(app_module, sistolica, categoria):
    flags = app_module.compute_clinical_flags(0, 0, sistolica)
    cats = [f["category"] for f in flags if f["indicator"] == "blood_pressure"]
    assert cats == ([categoria] if categoria else [])

def test_flags_citan_fuente(app_module):
    flags = app_module.compute_clinical_flags(250, 7.0, 185)
    assert {f["source"] for f in flags} == {"ADA", "ACC/AHA"}


# ---------- _safe_get (lookup tolerante de features en el payload) ----------

def test_safe_get_exacto(app_module):
    assert app_module._safe_get({"age": 40}, "age") == 40

def test_safe_get_case_insensitive(app_module):
    assert app_module._safe_get({"Age": 40}, "age") == 40

def test_safe_get_ausente(app_module):
    assert app_module._safe_get({"age": 40}, "bmi") is None
