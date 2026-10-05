"""Coherencia entre campos (auditoria 2026-08; revision 2026-10).

Combinaciones que no pueden ser de una misma persona aunque cada dato caiga en su
rango: una cintura que no cuadra con el IMC, una diastolica igual o mayor que la
sistolica, o un IMC que con ese peso implica una talla imposible. La v1 de hipertension
pedia peso, IMC y cintura como campos sueltos y su LogReg aprendio, por colinealidad,
un coeficiente NEGATIVO para el peso. Desde la v2 las reglas (coherence.py) miran los
campos que llegan, no la enfermedad, porque cada modo pide variables distintas.

Igual que la capa clinica (A4) y el soporte de datos (AUD-16), esto NO toca la
probabilidad: solo agrega un aviso mas a support_warnings.
"""
import numpy as np
import pandas as pd
import pytest

import coherence
import modos as M

COMPLETO = {
    "age": 45, "bmi": 25, "waist_circumference": 85, "ap_hi": 120, "ap_lo": 80,
    "total_cholesterol": 190, "hdl_cholesterol": 55, "hba1c_level": 5.4, "egfr": 95,
    "albumin_creatinine_ratio": 8, "diabetes": 0, "hypertension": 0, "heart_disease": 0,
    "high_cholesterol": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


def _completo(client, disease="diabetes", **cambios):
    r = client.post(f"/predict/{disease}?mode=completo", json={**COMPLETO, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _avisos(resp, level="incoherente"):
    return [a for a in resp["support_warnings"] if a["level"] == level]


# ---------- las reglas ----------

def test_las_reglas_solo_juzgan_lo_que_llega():
    assert coherence.incoherencias({}) == []
    assert coherence.incoherencias({"bmi": 19}) == []
    assert coherence.incoherencias({"ap_hi": 120, "ap_lo": 80, "bmi": 25, "waist_circumference": 85}) == []


def test_combinacion_coherente_no_avisa(client):
    assert _avisos(_completo(client)) == []


def test_cintura_enorme_con_imc_bajo_avisa(client):
    """El caso que motivo el chequeo: cintura enorme + IMC bajo es casi imposible."""
    r = _completo(client, bmi=19, waist_circumference=115)
    assert [a["feature"] for a in _avisos(r)] == ["waist_circumference"]
    assert "no es coherente" in r["support_note"]


def test_imc_alto_con_cintura_pequena_avisa(client):
    r = _completo(client, bmi=42, waist_circumference=70)
    assert [a["feature"] for a in _avisos(r)] == ["waist_circumference"]


# ---------- peso, talla e IMC ----------

def test_en_el_simplificado_peso_y_talla_cuadran_por_construccion(client, perfil_diabetes):
    """El IMC sale del peso y la talla: no hay talla implicita que juzgar."""
    r = client.post("/predict/diabetes", json=perfil_diabetes).get_json()
    assert _avisos(r) == []


def test_un_imc_dado_a_mano_que_no_cuadra_con_el_peso_avisa(client, perfil_diabetes):
    """180 kg con un IMC de 25 implican ~2,68 m de talla."""
    payload = {**perfil_diabetes, "bmi": 25, "weight": 180}
    r = client.post("/predict/diabetes", json=payload).get_json()
    assert [a["feature"] for a in _avisos(r)] == ["weight"]


def test_en_el_completo_el_peso_no_se_juzga(client):
    """El completo no pide el peso: si llega, no es dato del modo."""
    assert _avisos(_completo(client, weight=40, bmi=30)) == []   # 1,15 m implicitos


def test_la_ficha_real_que_manda_el_laboratorio_no_avisa(client):
    """El laboratorio evalua en el completo la ficha entera, con el peso DECLARADO y el
    IMC MEDIDO de la misma persona. Cruzar dos mediciones distintas le daba a gente
    real una talla implicita de menos de 1,30 m (9 de cada 9.500)."""
    tr = pd.read_csv("data_curated/diabetes/diabetes_train.csv")[M.columnas_laboratorio("diabetes")]
    tr = tr.dropna()
    ficha = tr[np.sqrt(tr["weight"] / tr["bmi"]) < 1.30].iloc[0].to_dict()
    r = client.post("/predict/diabetes?mode=completo", json=ficha)
    assert r.status_code == 200, r.get_json()
    assert _avisos(r.get_json()) == []


# ---------- presion: sistolica y diastolica (revision 2026-10) ----------
# Nada impedia mandar la presion al reves (100/120): cada valor cae en su rango y el
# modelo devolvia un numero como si nada.

@pytest.mark.parametrize("ap_hi,ap_lo", [(100, 120), (110, 110)])
@pytest.mark.parametrize("disease", ["cardiovascular", "hipertension"])
def test_diastolica_igual_o_mayor_que_la_sistolica_avisa(client, disease, ap_hi, ap_lo):
    """En hipertension la presion no entra al modelo, pero la lee la guia: tambien hay
    que avisar si llega al reves."""
    r = _completo(client, disease, ap_hi=ap_hi, ap_lo=ap_lo)
    incoherentes = _avisos(r)
    assert [a["feature"] for a in incoherentes] == ["ap_lo"]
    assert "intercambiadas" in incoherentes[0]["detail"]


def test_una_presion_normal_no_avisa(client):
    assert _avisos(_completo(client, "cardiovascular", ap_hi=90, ap_lo=85)) == []


# ---------- el aviso no toca el numero ----------

@pytest.mark.parametrize("cambios", [
    {"bmi": 19, "waist_circumference": 115}, {"ap_hi": 100, "ap_lo": 120}])
def test_el_aviso_no_toca_la_probabilidad(client, app_module, monkeypatch, cambios):
    """Mismo invariante que la capa clinica y AUD-16: el MISMO payload incoherente, con
    el chequeo activo y apagado, da exactamente la misma probabilidad."""
    con_aviso = _completo(client, **cambios)
    monkeypatch.setattr(app_module, "_check_coherencia", lambda *a: [])
    sin_aviso = _completo(client, **cambios)
    assert _avisos(con_aviso) and not _avisos(sin_aviso)
    assert con_aviso["probability"] == sin_aviso["probability"]
    assert con_aviso["raw_model_probability"] == sin_aviso["raw_model_probability"]
