"""Revision 2026-10 — en cardiovascular, fumar y beber no pueden bajar el riesgo.

Los habitos se dejaban sin restriccion de monotonia y el modelo aprendia que fumar y
beber protegen: sobre el test real, marcar "fumo" bajaba la probabilidad en el 59% de
los casos (-2,6 puntos de media) y "bebo", en el 75% (-4,1). No es fisiologia, es el
dataset: son autorreportados y ahi los fumadores enferman menos (47,4% frente a 50,0%).

Ahora `smoke` y `alco` no pueden bajar el riesgo y `active` no puede subirlo. Como los
datos tampoco dan senal en el sentido clinico, el efecto de fumar y beber queda en
CERO, y la capa clinica lo dice cuando el usuario los marca: callarlo seria otra forma
de mentir. Coste: AUC de test 0,7943 -> 0,7936.
"""
import pandas as pd
import pytest

from train_models import _monotone_vector

CARDIO = {
    "age": 55, "bmi": 28, "ap_hi": 135, "ap_lo": 85, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1, "gender_Female": 0, "gender_Male": 1,
}


@pytest.fixture(scope="module")
def test_real(app_module):
    feats = app_module.FEATURES["cardiovascular"]
    return pd.read_csv("data_curated/cardiovascular/cardiovascular_test.csv")[feats].astype(float)


def _cambio_en_cruda(app_module, X, feature, antes, despues):
    """Cuanto se mueve la salida del modelo al cambiar SOLO esa feature, fila a fila."""
    modelo = app_module.MODELS["cardiovascular"]
    base = X[X[feature] == antes]
    cambiado = base.copy()
    cambiado[feature] = despues
    assert len(base) > 1000
    return modelo.predict_proba(cambiado.values)[:, 1] - modelo.predict_proba(base.values)[:, 1]


# ---------- el modelo servido ----------

@pytest.mark.parametrize("habito", ["smoke", "alco"])
def test_fumar_y_beber_nunca_bajan_el_riesgo(app_module, test_real, habito):
    assert _cambio_en_cruda(app_module, test_real, habito, 0, 1).min() >= -1e-12


def test_ser_sedentario_nunca_baja_el_riesgo(app_module, test_real):
    cambio = _cambio_en_cruda(app_module, test_real, "active", 1, 0)
    assert cambio.min() >= -1e-12
    assert cambio.mean() > 0.01   # y este habito si pesa


@pytest.mark.parametrize("habito", ["smoke", "alco"])
def test_el_modelo_no_da_peso_a_fumar_ni_a_beber(app_module, test_real, habito):
    """Lo que afirman los textos de HABITOS_NO_REFLEJADOS. Si un reentrenamiento hace
    que estos habitos pesen, este test falla y hay que revisar esos textos."""
    assert abs(_cambio_en_cruda(app_module, test_real, habito, 0, 1)).max() <= 1e-12


def test_el_modelo_servido_lleva_las_restricciones_del_entrenamiento(app_module):
    feats = app_module.FEATURES["cardiovascular"]
    esperado = _monotone_vector("cardiovascular", feats)
    assert dict(zip(feats, esperado)) == {
        "age": 1, "bmi": 1, "ap_hi": 1, "ap_lo": 1, "cholesterol": 1, "gluc": 1,
        "smoke": 1, "alco": 1, "active": -1, "gender_Female": 0, "gender_Male": 0}
    clf = app_module.MODELS["cardiovascular"].named_steps["clf"]
    assert app_module.MODEL_NAMES["cardiovascular"] == "lightgbm"
    assert list(clf.get_params()["monotone_constraints"]) == esperado


# ---------- el aviso en la capa clinica ----------

def _predict(client, disease="cardiovascular", **cambios):
    r = client.post(f"/predict/{disease}", json={**CARDIO, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _habitos(resp):
    return [f["indicator"] for f in resp["clinical_flags"] if f["category"] == "no_reflejado_en_el_modelo"]


def test_quien_marca_que_fuma_se_entera_de_que_no_cuenta(client):
    d = _predict(client, smoke=1)
    assert _habitos(d) == ["smoke"]
    assert "fumas" in d["clinical_note"] and "no lo refleja" in d["clinical_note"]


def test_fumar_y_beber_se_avisan_por_separado(client):
    assert _habitos(_predict(client, smoke=1, alco=1)) == ["smoke", "alco"]
    assert _habitos(_predict(client, alco=1)) == ["alco"]


def test_sin_habitos_no_hay_aviso(client):
    assert _habitos(_predict(client)) == []
    assert _habitos(_predict(client, active=0)) == []   # el sedentarismo si lo refleja


def test_el_aviso_dice_la_verdad_la_estimacion_no_cambia(client):
    base = _predict(client)
    for cambios in ({"smoke": 1}, {"alco": 1}, {"smoke": 1, "alco": 1}):
        d = _predict(client, **cambios)
        assert d["probability"] == base["probability"]
        assert d["raw_model_probability"] == base["raw_model_probability"]


def test_el_aviso_es_solo_de_cardiovascular(client, perfil_diabetes):
    r = client.post("/predict/diabetes", json={**perfil_diabetes, "smoke": 1, "alco": 1}).get_json()
    assert _habitos(r) == []
