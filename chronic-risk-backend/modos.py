# modos.py
# Que variables usa cada modelo (dos modos por enfermedad), de que columna del dataset
# salen y en que sentido clinico pueden mover el riesgo. Lo comparten la ingesta, el
# entrenamiento (train_models.py), el sintetico y la app: por eso vive aparte y sin
# dependencias pesadas.
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ENFERMEDADES = ("diabetes", "hipertension", "cardiovascular", "renal", "higado")
MODOS = ("simplificado", "completo")

# El tabaco entra como dos columnas frente a "nunca ha fumado", que es la referencia:
# asi su signo SI se puede restringir (ver SENTIDO). El formulario sigue mandando las
# tres categorias.
COMUNES = ("age", "gender_Male", "gender_Female", "smoking_history_current",
           "smoking_history_former")

# Grupos que el usuario elige como una sola respuesta; la API exige una y solo una.
CATEGORICAS = {"gender": ("Female", "Male"), "smoking_history": ("current", "former", "never")}

# Simplificado: lo que cualquiera sabe de si mismo. Completo: lo que mide o pide el
# personal sanitario. Las variables que DEFINEN la enfermedad (HbA1c y glucosa en
# diabetes, presion en hipertension, filtrado y albumina en orina en la renal) se piden
# en el modo completo, pero ningun modelo las usa: las interpreta la capa clinica con
# las guias (ADA, OMS/ESC, KDIGO).
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
    # Enfermedad renal cronica. Sus factores de riesgo son los mismos que los de las
    # otras tres (la diabetes y la hipertension son sus dos primeras causas).
    "renal": {
        "simplificado": COMUNES + ("bmi", "diabetes", "hypertension", "high_cholesterol",
                                   "heart_disease"),
        "completo": COMUNES + ("bmi", "waist_circumference", "diabetes", "hypertension",
                               "high_cholesterol", "heart_disease", "ap_hi", "ap_lo",
                               "total_cholesterol", "hdl_cholesterol", "hba1c_level"),
    },
    # Higado graso (esteatosis hepatica por elastografia, CAP >= 288 dB/m). Lo mide un
    # FibroScan, que casi nadie tiene a mano: por eso el modelo es una estimacion a
    # partir de los factores metabolicos que la causan. La ALT (transaminasa) solo
    # entra en el completo; no define la enfermedad (la define el CAP, que no se pide).
    "higado": {
        "simplificado": COMUNES + ("bmi", "diabetes", "hypertension", "high_cholesterol",
                                   "heart_disease"),
        "completo": COMUNES + ("bmi", "waist_circumference", "diabetes", "hypertension",
                               "high_cholesterol", "heart_disease", "ap_hi", "ap_lo",
                               "total_cholesterol", "hdl_cholesterol", "hba1c_level", "alt"),
    },
}
DEFINITORIAS = {"diabetes": ("hba1c_level", "blood_glucose_level"),
                "hipertension": ("ap_hi", "ap_lo"),
                "cardiovascular": (),
                "renal": ("egfr", "albumin_creatinine_ratio"),
                "higado": ()}

# El modo simplificado lee el IMC AUTODECLARADO: es el que va a escribir el usuario,
# y la gente se quita IMC (-0,8 de media, -1,5 con obesidad y -2,3 con un IMC de 40 o
# mas).
FUENTE = {("simplificado", "bmi"): "bmi_autodeclarado"}

# En el simplificado el usuario no da su IMC sino su peso (kg) y su talla (cm): la API
# lo calcula. El completo pide el IMC medido, que es lo que tiene el personal sanitario.
DERIVADAS = {("simplificado", "bmi"): ("weight", "height")}


def columnas_laboratorio(enfermedad):
    """Lo que trae una ficha del laboratorio (real o sintetica): las variables del modo
    completo (incluyen las del simplificado), las que definen la enfermedad, el peso,
    la talla y el IMC autodeclarados, y el objetivo. Fuera la glucosa en ayunas, que
    solo tiene la mitad (submuestra de ayuno): el GAN trabaja con casos completos y se
    quedaria sin la otra mitad. Tampoco van el ciclo ni los pesos muestrales. Los
    grupos (sexo, tabaco) van con TODAS sus categorias: "nunca ha fumado" es una
    respuesta, aunque el modelo la use de referencia y no la lea."""
    grupos = [f"{g}_{o}" for g, opciones in CATEGORICAS.items() for o in opciones]
    cols = [c for c in FEATURES[enfermedad]["completo"] if not c.startswith(("gender_", "smoking_history_"))]
    cols += [c for c in DEFINITORIAS[enfermedad] if c != "blood_glucose_level"]
    return list(dict.fromkeys(["age"] + grupos + cols + ["weight", "height", "bmi_autodeclarado", "target"]))

# Sentido clinico: +1 = no puede bajar el riesgo, -1 = no puede subirlo, 0 = libre.
# El sexo queda libre. El tabaco no: fumar o haber fumado no puede bajar el riesgo
# frente a no haber fumado nunca. Sin la restriccion, en diabetes ser exfumador restaba
# 2,3 puntos (en datos de un solo momento, quien enferma deja de fumar).
SENTIDO = {
    "age": 1, "bmi": 1, "waist_circumference": 1, "hypertension": 1, "diabetes": 1,
    "high_cholesterol": 1, "heart_disease": 1, "ap_hi": 1, "ap_lo": 1,
    "total_cholesterol": 1, "hdl_cholesterol": -1, "hba1c_level": 1, "egfr": -1,
    "albumin_creatinine_ratio": 1, "alt": 1, "smoking_history_current": 1, "smoking_history_former": 1,
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
