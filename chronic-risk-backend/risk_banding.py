# risk_banding.py
# Regla de las bandas de riesgo bajo / moderado / alto. Vive en su propio modulo
# porque la usan dos sitios que no pueden importarse entre si: app.py, que la sirve,
# y los scripts de entrenamiento, que publican en *_metrics.json como reparte cada
# banda a la gente real del test (lo que /metricas ensena).
#
# El simulador pintaba las tres enfermedades por tercios fijos (33% y 66%). Con
# hipertension (36% de prevalencia) y cardiovascular (50%) tiene sentido; con
# diabetes (13,6%) no: una mujer sana de 25 anos con glucosa de 250 daba 31,8% —mas
# del doble que la media— y salia "Riesgo bajo", el 80% de los diabeticos reales del
# test caia en "bajo", y la variante sin glucosa, que no pasa de 54,5%, no podia
# llegar nunca a "alto".
# Regla: "bajo" es quedar por debajo de la media de los datos de entrenamiento y
# "alto", al menos el doble. Los tercios se quedan como tope, asi que solo cambian
# las enfermedades poco frecuentes.
from typing import Any, Dict, Optional

import numpy as np

LOW_CAP = 0.33
HIGH_CAP = 0.66
BANDS = ("low", "mid", "high")


def band_cutoffs(prevalence: Optional[float]) -> Dict[str, Any]:
    """Cortes para una enfermedad con esa prevalencia: por debajo de `low_below` es
    bajo; desde `high_from`, alto. Sin prevalencia quedan los tercios."""
    low, high = LOW_CAP, HIGH_CAP
    if prevalence is not None:
        low, high = min(low, prevalence), min(high, 2 * prevalence)
    return {
        "low_below": low, "high_from": high, "prevalence": prevalence,
        # False = se quedaron los tercios; True = los cortes salen de la prevalencia.
        "relative_to_prevalence": low < LOW_CAP or high < HIGH_CAP,
    }


def band_of(prob: float, cutoffs: Dict[str, Any]) -> str:
    if prob < cutoffs["low_below"]:
        return "low"
    return "mid" if prob < cutoffs["high_from"] else "high"


def band_report(y_true, proba, prevalence: Optional[float]) -> Dict[str, Any]:
    """Los cortes mas lo que hace cada banda sobre un conjunto etiquetado (el test):
    que parte de la gente cae en ella, cuantos de ellos tienen la enfermedad y que
    parte de todos los enfermos recoge. `proba` es la probabilidad CALIBRADA, la
    misma sobre la que se lee la banda al servir."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    cortes = band_cutoffs(prevalence)
    banda = np.array([band_of(p, cortes) for p in proba])
    total, enfermos = len(y_true), int(y_true.sum())
    filas = []
    for nombre in BANDS:
        dentro = banda == nombre
        n = int(dentro.sum())
        positivos = int(y_true[dentro].sum())
        filas.append({
            "band": nombre, "n": n,
            "share": n / total if total else 0.0,
            "positive_rate": positivos / n if n else None,
            "share_of_positives": positivos / enfermos if enfermos else None,
        })
    return {**cortes, "test": filas}
