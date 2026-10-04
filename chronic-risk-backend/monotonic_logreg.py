# monotonic_logreg.py
# Regresion logistica que respeta el sentido clinico declarado de cada feature
# (revision 2026-10). El porque esta en train_models.py, junto a MONOTONIC_INCREASING.
#
# Va en su propio modulo porque el pipeline guardado referencia esta clase: app.py
# tiene que poder importarla al cargarlo, sin arrastrar el entrenamiento.
import numpy as np
from sklearn.linear_model import LogisticRegression


class MonotonicLogisticRegression(LogisticRegression):
    """LogisticRegression con el signo de cada coeficiente acotado.

    `monotone_constraints` va alineado a las columnas, igual que en LightGBM: +1 = el
    coeficiente no puede ser negativo (la feature no puede bajar el riesgo), -1 = no
    puede ser positivo, 0 = libre. Si el ajuste deja alguno en contra, esa feature
    sale, se reajusta sin ella y al final vuelve con coeficiente 0 exacto.

    Con una sola feature en contra es el optimo exacto con la cota: la perdida es
    convexa, asi que ese optimo cae en el borde (coeficiente 0), y eso es ajustar sin
    ella. Con varias se sacan de una en una, empezando por la mas contraria.

    Sigue siendo una LogisticRegression: predict_proba, el calibrador y el
    LinearExplainer de SHAP la usan igual."""

    def __init__(self, monotone_constraints=None, *, C=1.0, class_weight=None, max_iter=100):
        super().__init__(C=C, class_weight=class_weight, max_iter=max_iter)
        self.monotone_constraints = monotone_constraints

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        signos = (np.zeros(n) if self.monotone_constraints is None
                  else np.asarray(self.monotone_constraints, dtype=float))
        if signos.shape != (n,):
            raise ValueError(f"monotone_constraints trae {signos.size} valores para {n} features")
        dentro = np.ones(n, dtype=bool)
        while True:
            super().fit(X[:, dentro], y, sample_weight=sample_weight)
            coef = np.zeros(n)
            coef[dentro] = self.coef_.ravel()
            en_contra = coef * signos
            if en_contra.min() >= 0:
                break
            dentro[np.argmin(en_contra)] = False
        self.coef_ = coef.reshape(1, -1)
        self.n_features_in_ = n
        return self
