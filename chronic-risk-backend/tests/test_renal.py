"""Enfermedad renal cronica, la cuarta enfermedad (revision 2026-10).

La define KDIGO por analitica: filtrado glomerular (eGFR) < 60 o albumina/creatinina en
orina >= 30 mg/g; cuenta tambien quien ya esta diagnosticado. En NHANES 2017-2023 la
tiene el 18,8% de los adultos con las dos analiticas, y solo el 19% de ellos lo sabia.
Como la HbA1c en diabetes, lo que la define no entra al modelo (reaprenderia el umbral):
se pide y lo interpreta la guia, sin tocar la probabilidad.
"""
import pytest

import modos as M

SIMPLE = {
    "age": 64, "weight": 95, "height": 170, "diabetes": 1, "hypertension": 1,
    "high_cholesterol": 0, "heart_disease": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
DEFINE = ["egfr", "albumin_creatinine_ratio"]


def _predict(client, **cambios):
    r = client.post("/predict/renal", json={**SIMPLE, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


# ---------- lo que la define lo lee la guia ----------

@pytest.mark.parametrize("modo", M.MODOS)
def test_lo_que_la_define_no_es_variable_del_modelo(client, modo):
    c = client.get(f"/config/renal?mode={modo}").get_json()
    assert c["defining_inputs"] == DEFINE
    assert not set(DEFINE) & set(c["features"])
    assert set(DEFINE) <= set(c["clinical_inputs"])


@pytest.mark.parametrize("disease", ["diabetes", "hipertension", "cardiovascular"])
def test_las_otras_no_piden_la_funcion_renal_en_el_simplificado(client, disease):
    """Un filtrado glomerular seria un campo mas que casi nadie conoce: el simplificado
    de las otras enfermedades sigue con sus cuatro opcionales."""
    c = client.get(f"/config/{disease}").get_json()
    assert c["optional_features"] == ["blood_glucose_level", "hba1c_level", "ap_hi", "ap_lo"]


def test_el_analisis_dispara_kdigo_sin_tocar_la_probabilidad(client):
    sin = _predict(client)
    con = _predict(client, egfr=52, albumin_creatinine_ratio=45)
    assert con["probability"] == sin["probability"]
    assert con["raw_model_probability"] == sin["raw_model_probability"]
    assert [f["category"] for f in con["clinical_flags"] if f["source"] == "KDIGO"] == [
        "filtrado_G3a", "albuminuria_A2"]
    assert not [f for f in sin["clinical_flags"] if f["source"] == "KDIGO"]


@pytest.mark.parametrize("feature", DEFINE)
def test_no_se_barre_lo_que_la_define(client, feature):
    r = client.post("/whatif/renal", json={"base": SIMPLE, "feature": feature, "min": 20, "max": 90})
    assert r.status_code == 400 and "recta" in r.get_json()["error"]


# ---------- el colesterol alto diagnosticado no tiene efecto ----------
# Quien lo tiene diagnosticado suele tomar estatinas: el modelo no encuentra senal en
# el sentido clinico y las restricciones lo dejan en cero.

def test_el_colesterol_alto_no_tiene_efecto(app_module):
    for modo in M.MODOS:
        assert "high_cholesterol" in app_module.SIN_EFECTO[f"renal_{modo}"]


def test_solo_avisa_si_se_marca(client):
    """Un si/no llega siempre, tambien con un "no": avisar entonces de que el modelo no
    lo refleja no tiene sentido."""
    no = _predict(client, high_cholesterol=0)
    si = _predict(client, high_cholesterol=1)
    assert si["probability"] == no["probability"]
    assert not [f for f in no["clinical_flags"] if f["indicator"] == "sin_efecto"]
    [aviso] = [f for f in si["clinical_flags"] if f["indicator"] == "sin_efecto"]
    assert aviso["value"] == ["high_cholesterol"] and "colesterol alto" in aviso["detail"]
