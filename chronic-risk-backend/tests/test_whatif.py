# Tests de /whatif (v2): curva contrafactual por modo, el peso del simplificado a talla
# fija, que solo se barra lo que el modelo usa, validaciones y no-logueo en BD.
import pytest

import modos as M


ENFERMEDADES = list(M.ENFERMEDADES)

# 160 cm: talla^2 = 2,56 m2 exactos, asi el IMC acoplado sale redondo.
BASE = {
    "age": 55, "weight": 80, "height": 160, "diabetes": 0, "hypertension": 0,
    "heart_disease": 0, "high_cholesterol": 1, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
COMPLETO = {
    "age": 55, "bmi": 31, "waist_circumference": 104, "ap_hi": 135, "ap_lo": 85,
    "total_cholesterol": 210, "hdl_cholesterol": 45, "hba1c_level": 5.8, "egfr": 85,
    "albumin_creatinine_ratio": 20, "diabetes": 0, "hypertension": 0, "heart_disease": 0,
    "high_cholesterol": 1, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


def _whatif(client, disease, body, modo=None):
    ruta = f"/whatif/{disease}" + (f"?mode={modo}" if modo else "")
    return client.post(ruta, json=body)


def _curva(client, disease, feature, vmin, vmax, steps, base=BASE, modo=None):
    r = _whatif(client, disease, {"base": base, "feature": feature, "min": vmin, "max": vmax,
                                  "steps": steps}, modo)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _no_baja(curva):
    for antes, despues in zip(curva, curva[1:]):
        assert despues["probability"] >= antes["probability"] - 1e-9
        assert despues["raw_probability"] >= antes["raw_probability"] - 1e-9


def test_whatif_no_loguea_en_bd(client, app_module):
    """El what-if es exploratorio: NO debe ensuciar la analitica del admin."""
    with app_module.SessionLocal() as s:
        antes = s.query(app_module.Prediction).count()
    _curva(client, "diabetes", "age", 20, 80, 10)
    with app_module.SessionLocal() as s:
        assert s.query(app_module.Prediction).count() == antes


def test_whatif_validaciones(client):
    assert _whatif(client, "diabetes", {"base": BASE, "min": 0, "max": 1}).status_code == 400
    assert _whatif(client, "diabetes", {"base": BASE, "feature": "age", "min": 40, "max": 20}).status_code == 400
    assert _whatif(client, "obesidad", {"base": {}, "feature": "age", "min": 20, "max": 40}).status_code == 404
    assert _whatif(client, "diabetes", {"base": BASE, "feature": "age", "min": 20, "max": 40},
                   modo="experto").status_code == 400


# ---------- el peso del simplificado se barre a talla fija (revision 2026-10) ----------
# Barrer el peso con el IMC quieto es barrer la ESTATURA: en la v1 la curva de
# hipertension bajaba (59% a 45 kg, 27% a 140 kg). En el simplificado el IMC sale del
# peso y la talla, asi que la talla del caso base se queda fija y el IMC sigue al peso.

@pytest.mark.parametrize("disease", ENFERMEDADES)
def test_mas_peso_no_baja_el_riesgo(client, disease):
    d = _curva(client, disease, "weight", 45, 140, 20)
    _no_baja(d["curve"])
    assert d["curve"][-1]["probability"] > d["curve"][0]["probability"]


def test_el_peso_mueve_el_imc_a_la_talla_del_caso_base(client):
    d = _curva(client, "hipertension", "weight", 48, 96, 4)
    assert d["coupled"] == {"feature": "bmi", "height_m": 1.6}
    assert [(p["value"], p["coupled_value"]) for p in d["curve"]] == [
        (48, 18.75), (64, 25), (80, 31.25), (96, 37.5)]


def test_cada_punto_es_la_prediccion_de_ese_peso(client):
    """La curva no inventa nada: es /predict con el peso barrido y la misma talla."""
    d = _curva(client, "hipertension", "weight", 48, 96, 4)
    for p in d["curve"]:
        directo = client.post("/predict/hipertension", json={**BASE, "weight": p["value"]}).get_json()
        assert p["raw_probability"] == pytest.approx(directo["raw_model_probability"], abs=1e-12)
        assert p["probability"] == pytest.approx(directo["probability"], abs=1e-12)


def test_barrer_el_imc_mueve_el_peso(client):
    d = _curva(client, "hipertension", "bmi", 20, 40, 5)
    assert d["coupled"] == {"feature": "weight", "height_m": 1.6}
    assert [(p["value"], p["coupled_value"]) for p in d["curve"]] == [
        (20, 51.2), (25, 64), (30, 76.8), (35, 89.6), (40, 102.4)]
    _no_baja(d["curve"])


def test_peso_e_imc_cuentan_la_misma_historia(client):
    """La misma persona a 96 kg (IMC 37,5) da el mismo riesgo se llegue por el barrido
    de peso o por el de IMC."""
    por_peso = _curva(client, "hipertension", "weight", 48, 96, 4)["curve"][-1]
    por_imc = _curva(client, "hipertension", "bmi", 25, 37.5, 3)["curve"][-1]
    assert por_peso["raw_probability"] == pytest.approx(por_imc["raw_probability"], abs=1e-12)


def test_barrer_el_peso_sin_talla_es_400(client):
    base = {k: v for k, v in BASE.items() if k != "height"}
    r = _whatif(client, "hipertension", {"base": base, "feature": "weight", "min": 50, "max": 90})
    assert r.status_code == 400 and "talla" in r.get_json()["error"]


def test_el_imc_acoplado_se_topa_a_los_limites_fisicos(client):
    """Talla de 2,30 m: a 45 kg el IMC derivado seria 8,5, por debajo del minimo que
    acepta la API (10). Un valor derivado no puede tumbar la curva con un 400."""
    d = _curva(client, "hipertension", "weight", 45, 140, 20, base={**BASE, "height": 230})
    assert min(p["coupled_value"] for p in d["curve"]) == 10


# ---------- lo que se puede barrer ----------

@pytest.mark.parametrize("disease,modo,feature", [
    ("diabetes", "completo", "hba1c_level"),          # define la diabetes: la lee la guia
    ("diabetes", None, "blood_glucose_level"),        # idem, y no es variable de ningun modelo
    ("hipertension", "completo", "ap_hi"),            # define la hipertension
    ("cardiovascular", "completo", "ap_hi"),          # sin efecto: las restricciones la anulan
    ("diabetes", None, "waist_circumference"),        # solo existe en el completo
])
def test_solo_se_barre_lo_que_el_modelo_usa(client, disease, modo, feature):
    base = COMPLETO if modo == "completo" else BASE
    r = _whatif(client, disease, {"base": base, "feature": feature, "min": 100, "max": 200}, modo)
    assert r.status_code == 400 and "recta" in r.get_json()["error"]


def test_el_completo_barre_sus_medidas(client):
    d = _curva(client, "diabetes", "albumin_creatinine_ratio", 0, 300, 13, base=COMPLETO, modo="completo")
    assert d["mode"] == "completo" and d["coupled"] is None
    _no_baja(d["curve"])
    assert d["supported_range"] is not None


def test_la_edad_marca_el_tope_de_nhanes(client):
    d = _curva(client, "diabetes", "age", 18, 90, 10)
    assert d["topcoded_at"] == 80
    assert d["supported_range"][1] == 80
