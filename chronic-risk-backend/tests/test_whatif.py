# Tests de /whatif: curva contrafactual, monotonia fina, ruteo y no-logueo en BD.


def _whatif(client, disease, body):
    return client.post(f"/whatif/{disease}", json=body)


def test_curva_glucosa_monotona(client, perfil_diabetes):
    """Barrido fino de glucosa: la curva calibrada Y la cruda deben ser
    no-decrecientes (protege el fix del 'pozo de glucosa', commit 7a6f86b)."""
    r = _whatif(client, "diabetes", {
        "base": perfil_diabetes, "feature": "blood_glucose_level",
        "min": 70, "max": 300, "steps": 40,
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d["variant"] == "glucosa"
    curva = d["curve"]
    assert len(curva) == 40
    for antes, despues in zip(curva, curva[1:]):
        assert despues["probability"] >= antes["probability"] - 1e-9
        assert despues["raw_probability"] >= antes["raw_probability"] - 1e-9


def test_whatif_no_loguea_en_bd(client, app_module, perfil_diabetes):
    """El what-if es exploratorio: NO debe ensuciar la analitica del admin."""
    with app_module.SessionLocal() as s:
        antes = s.query(app_module.Prediction).count()
    r = _whatif(client, "diabetes", {
        "base": perfil_diabetes, "feature": "bmi", "min": 20, "max": 40, "steps": 10,
    })
    assert r.status_code == 200
    with app_module.SessionLocal() as s:
        despues = s.query(app_module.Prediction).count()
    assert despues == antes


def test_whatif_sin_glucosa_usa_modelo_base(client, perfil_diabetes):
    r = _whatif(client, "diabetes", {
        "base": perfil_diabetes, "feature": "age", "min": 20, "max": 80, "steps": 5,
    })
    assert r.status_code == 200
    assert r.get_json()["variant"] == "base"


def test_whatif_validaciones(client, perfil_diabetes):
    # Falta 'feature'
    assert _whatif(client, "diabetes", {"base": perfil_diabetes, "min": 0, "max": 1}).status_code == 400
    # max <= min
    assert _whatif(client, "diabetes", {
        "base": perfil_diabetes, "feature": "bmi", "min": 40, "max": 20,
    }).status_code == 400
    # Enfermedad desconocida
    assert _whatif(client, "obesidad", {
        "base": {}, "feature": "bmi", "min": 20, "max": 40,
    }).status_code == 404
