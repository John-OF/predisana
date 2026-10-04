"""Revision 2026-10 — en hipertension, mas peso no puede bajar el riesgo.

La LogReg de hipertension gano el bake-off con el peso en -0,0167 por kg: a igual IMC y
cintura, mas peso es mas talla, y el modelo lo premiaba. Un hombre de 50 con IMC 30 y
cintura 100 daba 26,7% con 70 kg y 13,0% con 100 kg, y el SHAP ponia el peso en "lo
reduce" al 63% de la gente con obesidad del test. train_models.py ya declaraba que el
peso no puede bajar el riesgo, pero solo se lo pasaba a LightGBM.

Ahora la LogReg tambien respeta esas restricciones (MonotonicLogisticRegression) y el
peso queda en cero: su efecto llega por el IMC y la cintura. Coste: AUC de test
0,8037 -> 0,8027.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from monotonic_logreg import MonotonicLogisticRegression
from train_models import _monotone_vector

# El caso medido en la revision: hombre de 50 con IMC 30 y cintura 100.
HTA = {
    "age": 50, "bmi": 30, "weight": 85, "waist_circumference": 100,
    "diabetes": 0, "heart_disease": 0, "high_cholesterol": 0,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


@pytest.fixture(scope="module")
def test_real(app_module):
    feats = app_module.FEATURES["hipertension"]
    return pd.read_csv("data_curated/hipertension/hipertension_test.csv")[feats].astype(float)


# ---------- el modelo servido ----------

def test_el_modelo_servido_lleva_las_restricciones_del_entrenamiento(app_module):
    feats = app_module.FEATURES["hipertension"]
    esperado = _monotone_vector("hipertension", feats)
    assert dict(zip(feats, esperado)) == {
        "age": 1, "bmi": 1, "weight": 1, "waist_circumference": 1,
        "diabetes": 1, "heart_disease": 1, "high_cholesterol": 1,
        "gender_Male": 0, "gender_Female": 0, "smoking_history_never": 0,
        "smoking_history_current": 0, "smoking_history_former": 0}
    clf = app_module.MODELS["hipertension"].named_steps["clf"]
    assert app_module.MODEL_NAMES["hipertension"] == "logreg"
    assert list(clf.get_params()["monotone_constraints"]) == esperado
    assert all(c * s >= 0 for c, s in zip(clf.coef_[0], esperado))


def test_cambiar_solo_el_peso_no_mueve_el_riesgo(app_module, test_real):
    """Con el IMC y la cintura quietos, el peso no mueve nada: su efecto llega por el
    IMC. Antes, sumar peso BAJABA la salida del modelo en todo el test."""
    modelo = app_module.MODELS["hipertension"]
    mas_peso = test_real.copy()
    mas_peso["weight"] += 30
    assert np.array_equal(modelo.predict_proba(mas_peso.values),
                          modelo.predict_proba(test_real.values))


def test_el_shap_no_pone_el_peso_como_protector(app_module, test_real):
    scaler = app_module.MODELS["hipertension"].named_steps["scaler"]
    phi = np.array(app_module.EXPLAINERS["hipertension"].shap_values(
        scaler.transform(test_real.values)))
    assert np.abs(phi[:, app_module.FEATURES["hipertension"].index("weight")]).max() == 0


def test_el_caso_medido_ya_no_se_invierte(client):
    """Mismo IMC y misma cintura: 26,7% con 70 kg y 13,0% con 100 kg."""
    riesgos = {client.post("/predict/hipertension", json={**HTA, "weight": w})
               .get_json()["probability"] for w in (70, 85, 100, 115)}
    assert len(riesgos) == 1


def test_el_exceso_de_peso_se_explica_por_el_imc(client):
    r = client.post("/predict/hipertension", json={
        **HTA, "weight": 115, "bmi": 38, "waist_circumference": 120}).get_json()
    factores = {f["feature"]: f["shap"] for f in r["top_features"]}
    assert "weight" not in factores
    assert factores["bmi"] > 0 and factores["waist_circumference"] > 0


# ---------- MonotonicLogisticRegression ----------

def _datos(n=2000):
    """y sube con x0 y baja con x1; x2 es ruido."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] - X[:, 1] + rng.normal(size=n) > 0).astype(int)
    return X, y


def test_sin_restricciones_es_una_logreg_normal():
    X, y = _datos()
    m, ref = MonotonicLogisticRegression().fit(X, y), LogisticRegression().fit(X, y)
    assert np.array_equal(m.coef_, ref.coef_) and np.array_equal(m.intercept_, ref.intercept_)


def test_una_restriccion_que_ya_se_cumple_no_cambia_nada():
    X, y = _datos()
    assert np.array_equal(MonotonicLogisticRegression([1, -1, 0]).fit(X, y).coef_,
                          LogisticRegression().fit(X, y).coef_)


def test_la_feature_en_contra_queda_en_cero_y_el_resto_es_ajustar_sin_ella():
    X, y = _datos()
    m = MonotonicLogisticRegression([1, 1, 0]).fit(X, y)   # x1 baja y se le exige subir
    sin_x1 = LogisticRegression().fit(X[:, [0, 2]], y)
    assert m.coef_[0, 1] == 0
    assert np.array_equal(m.coef_[0, [0, 2]], sin_x1.coef_[0])
    assert np.array_equal(m.intercept_, sin_x1.intercept_)
    assert m.predict_proba(X).shape == (len(X), 2)


def test_se_clona_como_cualquier_estimador():
    """cross_val_score clona el estimador en cada fold del bake-off."""
    m = clone(MonotonicLogisticRegression([1, 0, -1], C=0.1, class_weight="balanced", max_iter=500))
    assert m.get_params() == {
        "monotone_constraints": [1, 0, -1], "C": 0.1, "class_weight": "balanced", "max_iter": 500}


def test_restricciones_de_otra_longitud_es_un_error():
    X, y = _datos()
    with pytest.raises(ValueError, match="3 features"):
        MonotonicLogisticRegression([1, 1]).fit(X, y)
