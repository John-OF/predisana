"""Revision 2026-10 — lo que la pagina de Educacion dice de cada modelo.

La caja "Variables Clave (IA)" de Educacion.jsx (`variables_ia`) describe los modelos
servidos, no la medicina en general. En la v1 no cuadraba (hablaba de la presion en
hipertension, que no era variable del modelo); con la v2 se reescribio con la media de
|SHAP| sobre el test real de cada modo. Si un reentrenamiento cambia el orden, estos
tests fallan y hay que revisar esos textos.
"""
import numpy as np
import pandas as pd
import pytest
import shap

import modos as M


def _importancia(app_module, clave):
    """Media de |SHAP| por variable sobre el test real, de mayor a menor. Los grupos
    (sexo, tabaco) se suman antes del valor absoluto."""
    disease, modo = app_module._partes(clave)
    feats = app_module.FEATURES[clave]
    te = pd.read_csv(f"data_curated/{disease}/{disease}_test.csv")
    X = pd.DataFrame({f: te[M.columna(modo, f)] for f in feats}).dropna().values.astype(float)
    Xt = app_module.MODELS[clave][:-1].transform(X)
    explainer = app_module.EXPLAINERS[clave]
    if isinstance(explainer, shap.TreeExplainer):
        sv = explainer.shap_values(Xt, check_additivity=False)
    else:
        sv = explainer.shap_values(Xt)
    sv = np.array(sv[-1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    por_variable = {}
    for i, f in enumerate(feats):
        nombre = "sexo" if f.startswith("gender_") else "tabaco" if f.startswith("smoking_") else f
        por_variable[nombre] = por_variable.get(nombre, 0) + sv[:, i]
    return pd.Series({k: np.abs(v).mean() for k, v in por_variable.items()}).sort_values(
        ascending=False)


@pytest.fixture(scope="module")
def importancia(app_module):
    return {clave: _importancia(app_module, clave) for clave in app_module.CLAVES}


# ---------- diabetes ----------

def test_diabetes_pesa_la_edad_luego_el_imc_y_la_presion_alta(importancia):
    assert list(importancia["diabetes_simplificado"].index[:3]) == ["age", "bmi", "hypertension"]


def test_diabetes_completo_suma_albumina_cintura_y_hdl(importancia):
    arriba = set(importancia["diabetes_completo"].index[:4])
    assert {"age", "albumin_creatinine_ratio", "waist_circumference", "hdl_cholesterol"} == arriba


# ---------- hipertension ----------

def test_hipertension_pesa_la_edad_luego_el_imc_y_el_colesterol(importancia):
    assert list(importancia["hipertension_simplificado"].index[:3]) == ["age", "bmi", "high_cholesterol"]


def test_hipertension_completo_suma_albumina_y_cintura(importancia):
    arriba = set(importancia["hipertension_completo"].index[:4])
    assert {"albumin_creatinine_ratio", "waist_circumference"} <= arriba


# ---------- cardiovascular ----------

def test_cardiovascular_pesa_la_edad_luego_la_presion_alta_y_el_tabaco(importancia):
    assert list(importancia["cardiovascular_simplificado"].index[:3]) == ["age", "hypertension", "tabaco"]


def test_cardiovascular_completo_no_usa_presion_colesterol_ni_hba1c(app_module, importancia):
    imp = importancia["cardiovascular_completo"]
    for f in ("ap_hi", "ap_lo", "total_cholesterol", "hba1c_level"):
        assert f in app_module.SIN_EFECTO["cardiovascular_completo"]
        assert imp[f] <= 1e-12, f


# ---------- lo que define la enfermedad no entra en el modelo ----------

def test_ni_la_hba1c_y_la_glucosa_en_diabetes_ni_la_presion_en_hipertension(app_module):
    for modo in M.MODOS:
        assert not {"hba1c_level", "blood_glucose_level"} & set(app_module.FEATURES[f"diabetes_{modo}"])
        assert not {"ap_hi", "ap_lo"} & set(app_module.FEATURES[f"hipertension_{modo}"])
