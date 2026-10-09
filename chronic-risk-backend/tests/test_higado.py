"""Higado graso, la quinta enfermedad (revision 2026-10).

El objetivo es MEDIDO, no autodeclarado: esteatosis por elastografia (FibroScan), CAP >=
288 dB/m en una exploracion valida de NHANES 2017-2023; la tiene el 33,5% de los adultos
explorados. Nada de lo que la define lo pide el formulario (casi nadie tiene un
FibroScan): el modelo la estima a partir de los factores metabolicos, y la ALT
(transaminasa) solo entra en el modo completo, porque es una variable del modelo y no la
que define la enfermedad.
"""
import pytest

import modos as M

SIMPLE = {
    "age": 52, "weight": 98, "height": 172, "diabetes": 1, "hypertension": 1,
    "high_cholesterol": 1, "heart_disease": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
COMPLETO = {k: v for k, v in SIMPLE.items() if k not in ("weight", "height")} | {
    "bmi": 33.1, "waist_circumference": 112, "ap_hi": 134, "ap_lo": 84,
    "total_cholesterol": 205, "hdl_cholesterol": 38, "hba1c_level": 6.1, "alt": 48}


def _predict(client, modo, **cambios):
    base = SIMPLE if modo == "simplificado" else COMPLETO
    r = client.post(f"/predict/higado?mode={modo}", json={**base, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_nada_define_el_higado_graso_en_el_formulario(client):
    for modo in M.MODOS:
        c = client.get(f"/config/higado?mode={modo}").get_json()
        assert c["defining_inputs"] == []


def test_la_alt_solo_es_variable_del_modo_completo(client):
    assert "alt" not in client.get("/config/higado?mode=simplificado").get_json()["features"]
    assert "alt" in client.get("/config/higado?mode=completo").get_json()["features"]


def test_el_completo_pide_la_alt(client):
    r = client.post("/predict/higado?mode=completo",
                    json={k: v for k, v in COMPLETO.items() if k != "alt"})
    assert r.status_code == 400 and "alt" in r.get_json()["error"]


@pytest.mark.parametrize("alt", [0, 5000])
def test_la_alt_fuera_de_limites_fisicos_se_rechaza(client, alt):
    r = client.post("/predict/higado?mode=completo", json={**COMPLETO, "alt": alt})
    assert r.status_code == 400


def test_mas_alt_nunca_baja_el_riesgo(client):
    p = [_predict(client, "completo", alt=a)["probability"] for a in (10, 20, 40, 80, 160)]
    assert p == sorted(p) and p[-1] > p[0]


def test_mas_peso_no_baja_el_riesgo_en_el_simplificado(client):
    p = [_predict(client, "simplificado", weight=w)["probability"] for w in (60, 80, 100, 120)]
    assert p == sorted(p) and p[-1] > p[0]


def test_la_cardiopatia_previa_no_cambia_la_estimacion_y_se_avisa(client):
    for modo in M.MODOS:
        no = _predict(client, modo, heart_disease=0)
        si = _predict(client, modo, heart_disease=1)
        assert si["probability"] == no["probability"]
        [aviso] = [f for f in si["clinical_flags"] if f["indicator"] == "sin_efecto"
                   and "heart_disease" in f["value"]]
        assert "cardiovascular" in aviso["detail"]
        assert not [f for f in no["clinical_flags"] if f["indicator"] == "sin_efecto"
                    and "heart_disease" in f["value"]]
