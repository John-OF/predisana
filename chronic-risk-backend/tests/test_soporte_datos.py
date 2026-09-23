"""AUD-16 — aviso de poco soporte de datos.

Un modelo no avisa de que está extrapolando: devuelve un número igual de firme para
una edad que vio 5000 veces que para una que no vio nunca. Cardiovascular se entrenó
con edades 29,7-64,9 y el formulario acepta 18-100, así que a los 90 años la respuesta
era una extrapolación presentada como una estimación normal.

Igual que la capa clínica (A4), esto se calcula APARTE: no puede tocar la probabilidad.
"""
CARDIO = {
    "age": 50, "bmi": 26, "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1, "gender_Female": 1, "gender_Male": 0,
}


def _predict(client, **cambios):
    return client.post("/predict/cardiovascular", json={**CARDIO, **cambios}).get_json()


def _avisos(resp, feature):
    return [a for a in resp["support_warnings"] if a["feature"] == feature]


# ---------- estadisticas de cobertura ----------
def test_config_expone_el_rango_entrenado(client):
    sup = client.get("/config/cardiovascular").get_json()["feature_support"]
    assert 29 <= sup["age"]["min"] <= 31 and 64 <= sup["age"]["max"] <= 66
    assert sup["age"]["p1"] > sup["age"]["min"] and sup["age"]["p99"] < sup["age"]["max"]
    assert sup["age"]["n"] > 1000


def test_las_categoricas_no_tienen_rango_de_soporte(client):
    """Un one-hot no tiene 'zona con pocos datos': el aviso no aplica."""
    sup = client.get("/config/cardiovascular").get_json()["feature_support"]
    assert "gender_Male" not in sup
    assert "smoke" not in sup


# ---------- los avisos ----------
def test_valor_central_no_avisa(client):
    r = _predict(client)
    assert r["support_warnings"] == []
    assert "dentro del rango" in r["support_note"]


def test_fuera_del_rango_entrenado_avisa_de_extrapolacion(client):
    """El caso de AUD-16: cardio no vio a nadie de 90 años."""
    r = _predict(client, age=90)
    avisos = _avisos(r, "age")
    assert len(avisos) == 1 and avisos[0]["level"] == "sin_datos"
    lo, hi = avisos[0]["trained_range"]
    assert lo <= 30 and hi <= 66
    assert "extrapolación" in r["support_note"]


def test_la_cola_avisa_mas_suave(client):
    """35 años está dentro del rango pero por debajo del p1: menos fiable, no ausente."""
    r = _predict(client, age=35)
    avisos = _avisos(r, "age")
    assert len(avisos) == 1 and avisos[0]["level"] == "pocos_datos"
    assert "extrapolación" not in r["support_note"]


def test_el_aviso_no_toca_la_probabilidad(client, app_module, monkeypatch):
    """Lo mismo que se exige a la capa clínica: informar sin alterar el número.
    El MISMO caso a los 90 años, con el aviso activo y con el aviso apagado, tiene
    que dar exactamente la misma probabilidad (calibrada y cruda)."""
    r90 = _predict(client, age=90)
    assert r90["support_warnings"]

    monkeypatch.setattr(app_module, "compute_support_warnings", lambda *a: [])
    sin_aviso = _predict(client, age=90)
    assert sin_aviso["support_warnings"] == []
    assert r90["probability"] == sin_aviso["probability"]
    assert r90["raw_model_probability"] == sin_aviso["raw_model_probability"]

    r50 = _predict(client)
    assert r90["probability"] > r50["probability"]   # sigue siendo monotona en edad


def test_un_campo_vacio_no_genera_aviso_de_soporte(client, perfil_diabetes):
    """Un campo sin rellenar ya se reporta por `missing_filled_as_zero`; duplicarlo
    como 'fuera de rango' seria ruido."""
    sin_bmi = {k: v for k, v in perfil_diabetes.items() if k != "bmi"}
    r = client.post("/predict/diabetes", json=sin_bmi).get_json()
    assert "bmi" in r["missing_filled_as_zero"]
    assert not _avisos(r, "bmi")


def test_whatif_devuelve_el_rango_soportado(client):
    """La curva se barre mas alla de lo entrenado y se aplana por falta de datos, no
    porque el riesgo deje de subir: el front sombrea esa zona."""
    r = client.post("/whatif/cardiovascular", json={
        "base": CARDIO, "feature": "age", "min": 18, "max": 100, "steps": 10,
    }).get_json()
    lo, hi = r["supported_range"]
    assert 18 < lo < 40 and 60 < hi < 70


def test_whatif_sin_rango_para_una_categorica(client):
    r = client.post("/whatif/cardiovascular", json={
        "base": CARDIO, "feature": "smoke", "min": 0, "max": 1, "steps": 2,
    }).get_json()
    assert r["supported_range"] is None
