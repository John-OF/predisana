# Tests de /whatif: curva contrafactual, monotonia fina, ruteo y no-logueo en BD.
import pytest


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


# ---------- hipertension: peso e IMC se barren a talla fija (revision 2026-10) ----------
# Barrer el peso con el IMC quieto es barrer la ESTATURA, y con el coeficiente negativo
# de weight la curva bajaba: 59% a 45 kg, 27% a 140 kg. Ahora la talla del caso base
# (peso / IMC = talla^2) se queda fija y el otro campo sigue al que se barre.

# 90 kg con IMC 30 -> talla^2 = 3.0 m2 exactos (1,73 m).
HTA = {
    "age": 55, "bmi": 30, "weight": 90, "waist_circumference": 104,
    "diabetes": 0, "heart_disease": 0, "high_cholesterol": 1,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


def _curva_hta(client, feature, vmin, vmax, steps, base=HTA):
    r = _whatif(client, "hipertension", {
        "base": base, "feature": feature, "min": vmin, "max": vmax, "steps": steps,
    })
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_mas_peso_no_baja_el_riesgo_de_hipertension(client):
    d = _curva_hta(client, "weight", 45, 140, 20)
    curva = d["curve"]
    for antes, despues in zip(curva, curva[1:]):
        assert despues["probability"] >= antes["probability"] - 1e-9
        assert despues["raw_probability"] >= antes["raw_probability"] - 1e-9
    assert curva[-1]["probability"] > curva[0]["probability"]


def test_el_peso_arrastra_al_imc_a_la_talla_del_caso_base(client):
    d = _curva_hta(client, "weight", 60, 120, 4)
    assert d["coupled"] == {"feature": "bmi", "height_m": 1.73}
    assert [(p["value"], p["coupled_value"]) for p in d["curve"]] == [
        (60, 20), (80, 26.67), (100, 33.33), (120, 40)]


def test_cada_punto_es_la_prediccion_de_ese_peso_con_su_imc(client):
    """La curva no inventa nada: es /predict con el peso barrido y el IMC que le toca."""
    d = _curva_hta(client, "weight", 60, 120, 4)
    for p in d["curve"]:
        directo = client.post("/predict/hipertension", json={
            **HTA, "weight": p["value"], "bmi": p["value"] / 3.0}).get_json()
        assert p["raw_probability"] == pytest.approx(directo["raw_model_probability"], abs=1e-12)
        assert p["probability"] == pytest.approx(directo["probability"], abs=1e-12)


def test_barrer_el_imc_arrastra_al_peso(client):
    d = _curva_hta(client, "bmi", 20, 40, 5)
    assert d["coupled"] == {"feature": "weight", "height_m": 1.73}
    assert [(p["value"], p["coupled_value"]) for p in d["curve"]] == [
        (20, 60), (25, 75), (30, 90), (35, 105), (40, 120)]
    for antes, despues in zip(d["curve"], d["curve"][1:]):
        assert despues["probability"] >= antes["probability"] - 1e-9


def test_peso_e_imc_cuentan_la_misma_historia(client):
    """La misma persona a 120 kg (IMC 40) da el mismo riesgo se llegue por el barrido
    de peso o por el de IMC. Antes uno subia y el otro bajaba."""
    por_peso = _curva_hta(client, "weight", 60, 120, 4)["curve"][-1]
    por_imc = _curva_hta(client, "bmi", 20, 40, 5)["curve"][-1]
    assert por_peso["raw_probability"] == pytest.approx(por_imc["raw_probability"], abs=1e-12)


def test_sin_peso_o_imc_en_el_caso_base_no_hay_acople(client):
    """Sin los dos no hay talla que deducir: se barre la feature sola, como antes."""
    base = {k: v for k, v in HTA.items() if k != "bmi"}
    d = _curva_hta(client, "weight", 60, 120, 4, base=base)
    assert d["coupled"] is None
    assert all("coupled_value" not in p for p in d["curve"])


def test_el_valor_acoplado_se_topa_a_los_limites_fisicos(client):
    """Talla de 2,20 m: a 45 kg el IMC derivado seria 9,3, por debajo del minimo que
    acepta la API (10). Un valor derivado no puede tumbar la curva con un 400."""
    base = {**HTA, "weight": 60, "bmi": 12.4}
    d = _curva_hta(client, "weight", 45, 140, 20, base=base)
    assert min(p["coupled_value"] for p in d["curve"]) == 10


def test_el_acople_es_solo_de_hipertension(client, perfil_diabetes):
    """El resto de barridos (cintura y edad incluidas) siguen moviendo una sola."""
    assert _curva_hta(client, "waist_circumference", 70, 130, 4)["coupled"] is None
    assert _curva_hta(client, "age", 30, 70, 4)["coupled"] is None
    r = _whatif(client, "diabetes", {
        "base": perfil_diabetes, "feature": "bmi", "min": 20, "max": 40, "steps": 5,
    })
    assert r.get_json()["coupled"] is None
