"""Revision 2026-10 — lo que la pagina de Educacion dice de cada modelo.

La caja "Variables Clave (IA)" de Educacion.jsx (`variables_ia`) no cuadraba con los
modelos servidos: en hipertension decia que lo que mas pesa es la presion, que no es
feature del modelo (AUD-1); en diabetes nombraba la HbA1c, que no usa ningun modelo;
y en cardiovascular, el tabaco, que pesa cero. Los textos se reescribieron con la
media de |SHAP| sobre el test real. Si un reentrenamiento cambia el orden, estos tests
fallan y hay que revisar esos textos.
"""
import numpy as np
import pandas as pd
import pytest
import shap

GRUPOS_ONE_HOT = ("gender_", "smoking_history_")


def _importancia(app_module, key, datos):
    """Media de |SHAP| por variable sobre el test real, de mayor a menor. Las one-hot
    se suman por grupo (sexo, tabaquismo) antes del valor absoluto."""
    feats = app_module.FEATURES[key]
    df = pd.read_csv(f"data_curated/{datos}/{datos}_test.csv", low_memory=False)
    X = df.reindex(columns=feats, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0)
    if "blood_glucose_level" in feats:
        X = X[X["blood_glucose_level"] > 0]   # la variante con glucosa solo se sirve con ella
    Xt = app_module.MODELS[key].named_steps["scaler"].transform(X.values.astype(float))
    explainer = app_module.EXPLAINERS[key]
    if isinstance(explainer, shap.TreeExplainer):
        sv = explainer.shap_values(Xt, check_additivity=False)
    else:
        sv = explainer.shap_values(Xt)
    sv = np.array(sv[-1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    por_variable = {}
    for i, f in enumerate(feats):
        nombre = next((g.rstrip("_") for g in GRUPOS_ONE_HOT if f.startswith(g)), f)
        por_variable[nombre] = por_variable.get(nombre, 0) + sv[:, i]
    return pd.Series({k: np.abs(v).mean() for k, v in por_variable.items()}).sort_values(
        ascending=False)


@pytest.fixture(scope="module")
def importancia(app_module):
    modelos = [("diabetes", "diabetes"), (app_module.DIABETES_GLUCOSE_KEY, "diabetes"),
               ("hipertension", "hipertension"), ("cardiovascular", "cardiovascular")]
    return {key: _importancia(app_module, key, datos) for key, datos in modelos}


def test_diabetes_sin_analisis_pesa_la_edad_luego_la_hipertension_y_el_imc(importancia):
    assert list(importancia["diabetes"].index[:3]) == ["age", "hypertension", "bmi"]


def test_diabetes_con_glucosa_la_glucosa_es_lo_que_mas_pesa(app_module, importancia):
    assert importancia[app_module.DIABETES_GLUCOSE_KEY].index[0] == "blood_glucose_level"


def test_hipertension_pesa_la_edad_luego_el_colesterol_y_el_imc(importancia):
    assert list(importancia["hipertension"].index[:3]) == ["age", "high_cholesterol", "bmi"]


def test_cardiovascular_la_sistolica_con_diferencia_luego_la_edad_y_el_colesterol(importancia):
    imp = importancia["cardiovascular"]
    assert list(imp.index[:3]) == ["ap_hi", "age", "cholesterol"]
    assert imp["ap_hi"] > 2 * imp["age"]
    assert imp["smoke"] <= 1e-12 and imp["alco"] <= 1e-12


def test_ni_la_presion_en_hipertension_ni_la_hba1c_entran_en_un_modelo(app_module):
    assert not {"blood_pressure", "ap_hi", "ap_lo"} & set(app_module.FEATURES["hipertension"])
    for key, feats in app_module.FEATURES.items():
        assert "hba1c_level" not in feats, key
