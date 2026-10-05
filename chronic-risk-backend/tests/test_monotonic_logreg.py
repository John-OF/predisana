"""MonotonicLogisticRegression (revision 2026-10): una LogReg con el signo de cada
coeficiente acotado.

Nacio con el peso de la v1 de hipertension: la LogReg gano el bake-off con el peso en
-0,0167 por kg (a igual IMC y cintura, mas peso es mas talla) y el modelo lo premiaba.
train_models.py declaraba que el peso no puede bajar el riesgo, pero solo se lo pasaba
a LightGBM. Desde entonces la LogReg candidata respeta las mismas restricciones: la
feature que sale al reves se reajusta fuera y vuelve con coeficiente 0 exacto.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from monotonic_logreg import MonotonicLogisticRegression


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
