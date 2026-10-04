# modos_v2.py
# Que variables usa cada modelo de la v2, de que columna del dataset salen y en que
# sentido clinico pueden mover el riesgo. Lo comparten el entrenamiento (train_v2.py)
# y, cuando se cambie, la app: por eso vive aparte y sin dependencias pesadas.
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ENFERMEDADES = ("diabetes", "hipertension", "cardiovascular")
MODOS = ("simplificado", "completo")

COMUNES = ("age", "gender_Male", "gender_Female", "smoking_history_never",
           "smoking_history_current", "smoking_history_former")

# Simplificado: lo que cualquiera sabe de si mismo. Completo: lo que mide o pide el
# personal sanitario. Las variables que DEFINEN la enfermedad (HbA1c y glucosa en
# diabetes, presion en hipertension) se piden en el modo completo, pero ningun modelo
# las usa: las interpreta la capa clinica con las guias (ADA, OMS/ESC).
FEATURES = {
    "diabetes": {
        "simplificado": COMUNES + ("bmi", "hypertension", "high_cholesterol", "heart_disease"),
        "completo": COMUNES + ("bmi", "waist_circumference", "hypertension", "high_cholesterol",
                               "heart_disease", "ap_hi", "ap_lo", "total_cholesterol",
                               "hdl_cholesterol", "egfr", "albumin_creatinine_ratio"),
    },
    "hipertension": {
        "simplificado": COMUNES + ("bmi", "diabetes", "high_cholesterol", "heart_disease"),
        "completo": COMUNES + ("bmi", "waist_circumference", "diabetes", "high_cholesterol",
                               "heart_disease", "total_cholesterol", "hdl_cholesterol",
                               "hba1c_level", "egfr", "albumin_creatinine_ratio"),
    },
    "cardiovascular": {
        "simplificado": COMUNES + ("bmi", "diabetes", "hypertension", "high_cholesterol"),
        "completo": COMUNES + ("bmi", "waist_circumference", "diabetes", "hypertension",
                               "high_cholesterol", "ap_hi", "ap_lo", "total_cholesterol",
                               "hdl_cholesterol", "hba1c_level", "egfr", "albumin_creatinine_ratio"),
    },
}
DEFINITORIAS = {"diabetes": ("hba1c_level", "blood_glucose_level"),
                "hipertension": ("ap_hi", "ap_lo"),
                "cardiovascular": ()}

# El modo simplificado lee el IMC AUTODECLARADO: es el que va a escribir el usuario,
# y la gente se quita IMC (-0,8 de media, -1,9 con obesidad).
FUENTE = {("simplificado", "bmi"): "bmi_autodeclarado"}

# Sentido clinico: +1 = no puede bajar el riesgo, -1 = no puede subirlo, 0 = libre.
# Sexo y tabaco quedan libres: son grupos one-hot (el coeficiente de cada columna es
# relativo a las otras) y su signo en datos de un solo momento no es inequivoco.
SENTIDO = {
    "age": 1, "bmi": 1, "waist_circumference": 1, "hypertension": 1, "diabetes": 1,
    "high_cholesterol": 1, "heart_disease": 1, "ap_hi": 1, "ap_lo": 1,
    "total_cholesterol": 1, "hdl_cholesterol": -1, "hba1c_level": 1, "egfr": -1,
    "albumin_creatinine_ratio": 1,
}

# Variables muy asimetricas: el modelo las ve en escala log (1 + x). Es monotona, asi
# que no cambia el sentido ni el orden; la API recibe el valor tal cual.
LOG = ("albumin_creatinine_ratio",)


def features(enfermedad, modo):
    return list(FEATURES[enfermedad][modo])


def columna(modo, feature):
    """Columna del dataset v2 de la que sale una feature."""
    return FUENTE.get((modo, feature), feature)


def sentidos(enfermedad, modo):
    return [SENTIDO.get(f, 0) for f in features(enfermedad, modo)]


class LogColumnas(BaseEstimator, TransformerMixin):
    """log(1 + x) en las columnas indicadas; el resto pasa igual. Va dentro del
    pipeline para que el modelo reciba los valores tal como los escribe la gente."""

    def __init__(self, columnas=()):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if len(self.columnas):
            X[:, list(self.columnas)] = np.log1p(np.clip(X[:, list(self.columnas)], 0, None))
        return X


def columnas_log(enfermedad, modo):
    return tuple(i for i, f in enumerate(features(enfermedad, modo)) if f in LOG)
