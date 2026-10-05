"""AUD-16 — aviso de poco soporte de datos.

Un modelo no avisa de que está extrapolando: devuelve un número igual de firme para
un valor que vio miles de veces que para uno que no vio nunca. Cada modo tiene su
propio rango (el simplificado ve el IMC declarado; el completo, el medido, que llega
más alto). Igual que la capa clínica (A4), esto se calcula APARTE: no puede tocar la
probabilidad.
"""
import pandas as pd
import pytest

import modos as M

SIMPLE = {
    "age": 50, "weight": 70, "height": 170, "diabetes": 0, "hypertension": 0,
    "high_cholesterol": 0, "gender_Female": 1, "gender_Male": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
COMPLETO = {
    **{k: v for k, v in SIMPLE.items() if k not in ("weight", "height")},
    "bmi": 24, "waist_circumference": 88, "ap_hi": 120, "ap_lo": 78, "total_cholesterol": 190,
    "hdl_cholesterol": 55, "hba1c_level": 5.4, "egfr": 95, "albumin_creatinine_ratio": 8,
}


def _predict(client, modo="simplificado", **cambios):
    base = SIMPLE if modo == "simplificado" else COMPLETO
    r = client.post(f"/predict/cardiovascular?mode={modo}", json={**base, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _avisos(resp, feature):
    return [a for a in resp["support_warnings"] if a["feature"] == feature]


def _soporte(client, modo="simplificado"):
    return client.get(f"/config/cardiovascular?mode={modo}").get_json()["feature_support"]


# ---------- estadisticas de cobertura ----------

def test_config_expone_el_rango_entrenado(client):
    imc = _soporte(client)["bmi"]
    assert imc["min"] < imc["p1"] < imc["p99"] < imc["max"]
    assert imc["n"] > 1000
    # El peso y la talla que pide el formulario tambien tienen su rango.
    assert {"weight", "height"} <= set(_soporte(client))


@pytest.mark.parametrize("modo,columna", [("simplificado", "bmi_autodeclarado"), ("completo", "bmi")])
def test_cada_modo_tiene_su_propio_rango(client, modo, columna):
    """El simplificado se entreno con el IMC declarado; el completo, con el medido y solo
    con quien tiene sus analisis. El rango de cada uno sale de sus filas."""
    tr = pd.read_csv("data_curated/cardiovascular/cardiovascular_train.csv")
    filas = tr.dropna(subset=[M.columna(modo, f) for f in M.features("cardiovascular", modo)])
    assert _soporte(client, modo)["bmi"]["max"] == filas[columna].max()
    assert len(filas) == _soporte(client, modo)["bmi"]["n"]


def test_las_categoricas_no_tienen_rango_de_soporte(client):
    """Un one-hot o un si/no no tiene 'zona con pocos datos': el aviso no aplica."""
    sup = _soporte(client)
    assert "gender_Male" not in sup and "hypertension" not in sup


# ---------- los avisos ----------

def test_valor_central_no_avisa(client):
    r = _predict(client)
    assert r["support_warnings"] == []
    assert "dentro del rango" in r["support_note"]


def test_fuera_del_rango_entrenado_avisa_de_extrapolacion(client):
    r = _predict(client, "completo", egfr=190)
    avisos = _avisos(r, "egfr")
    assert len(avisos) == 1 and avisos[0]["level"] == "sin_datos"
    assert avisos[0]["trained_range"][1] < 190
    assert "extrapolación" in r["support_note"]


def test_la_cola_avisa_mas_suave(client):
    """Un IMC de ~60 esta dentro de lo visto pero por encima del p99: menos fiable."""
    r = _predict(client, weight=175, height=170)
    avisos = _avisos(r, "bmi")
    assert len(avisos) == 1 and avisos[0]["level"] == "pocos_datos"
    assert "extrapolación" not in r["support_note"]


def test_por_encima_del_tope_de_nhanes(client):
    """NHANES registra como 80 a todo el que pasa de 80: si vio casos, agrupados."""
    avisos = _avisos(_predict(client, age=90), "age")
    assert avisos[0]["topcoded"] == 80 and "figura como 80" in avisos[0]["detail"]


def test_lo_que_el_modelo_no_usa_no_avisa(client, app_module):
    """Las variables sin efecto no mueven la estimacion: no hay extrapolacion que avisar."""
    assert "ap_hi" in app_module.SIN_EFECTO["cardiovascular_completo"]
    assert not _avisos(_predict(client, "completo", ap_hi=250, ap_lo=150), "ap_hi")


@pytest.mark.parametrize("cambios", [{"age": 90}, {"weight": 175}])
def test_el_aviso_no_toca_la_probabilidad(client, app_module, monkeypatch, cambios):
    """Lo mismo que se exige a la capa clínica: informar sin alterar el número."""
    con_aviso = _predict(client, **cambios)
    assert con_aviso["support_warnings"]
    monkeypatch.setattr(app_module, "compute_support_warnings", lambda *a: [])
    sin_aviso = _predict(client, **cambios)
    assert sin_aviso["support_warnings"] == []
    assert con_aviso["probability"] == sin_aviso["probability"]
    assert con_aviso["raw_model_probability"] == sin_aviso["raw_model_probability"]


def test_whatif_devuelve_el_rango_soportado(client):
    """La curva se barre mas alla de lo entrenado y se aplana por falta de datos, no
    porque el riesgo deje de subir: el front sombrea esa zona."""
    r = client.post("/whatif/cardiovascular", json={
        "base": SIMPLE, "feature": "age", "min": 18, "max": 100, "steps": 10}).get_json()
    assert r["supported_range"] == [20, 80]
