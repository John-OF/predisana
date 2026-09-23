"""Auditoria 2026-08 — coherencia weight / bmi / waist_circumference en hipertension.

Hipertension pide weight, bmi y waist_circumference como 3 campos SUELTOS (sin
altura, sin validacion cruzada). En los datos reales estan fuertemente
correlacionados (r~0.89-0.90); el LogReg ganador del bake-off aprendio, por
colinealidad, un coeficiente NEGATIVO para 'weight' (el unico signo invertido
entre las features con direccion clinica inequivoca): a igual bmi/cintura, MAS
peso puede dar MENOS riesgo. Con datos que covarian de forma realista el modelo
predice bien; el sintoma solo aparece con combinaciones incoherentes que antes
pasaban sin aviso porque cada campo, por separado, cae dentro de su rango.

Igual que la capa clinica (A4) y el soporte de datos (AUD-16), esto NO toca la
probabilidad: solo agrega un aviso mas a support_warnings.
"""
import pytest

HTA = {
    "age": 45, "bmi": 25, "weight": 75, "waist_circumference": 85,
    "diabetes": 0, "heart_disease": 0, "high_cholesterol": 0,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


def _predict(client, **cambios):
    return client.post("/predict/hipertension", json={**HTA, **cambios}).get_json()


def _avisos(resp, level=None):
    ws = resp["support_warnings"]
    return [a for a in ws if level is None or a["level"] == level]


def test_combinacion_coherente_no_avisa(client):
    """bmi=25, weight=75, waist=85 implican una altura (~1.73m) plausible."""
    r = _predict(client)
    assert _avisos(r, "incoherente") == []


def test_cintura_enorme_con_imc_bajo_avisa(client):
    """El caso que motivo el chequeo: cintura enorme + IMC bajo es casi imposible."""
    r = _predict(client, bmi=19, waist_circumference=115, weight=55)
    incoherentes = _avisos(r, "incoherente")
    assert any(a["feature"] == "waist_circumference" for a in incoherentes)
    assert "incoherente" in r["support_note"] or "no cuadran" in r["support_note"]


def test_imc_alto_con_cintura_pequena_avisa(client):
    r = _predict(client, bmi=42, waist_circumference=70, weight=110)
    incoherentes = _avisos(r, "incoherente")
    assert any(a["feature"] == "waist_circumference" for a in incoherentes)


def test_peso_e_imc_implican_altura_absurda(client):
    """weight=180 con bmi=25 implica ~2.68m de altura: fuera de rango humano plausible."""
    r = _predict(client, weight=180, bmi=25, waist_circumference=85)
    incoherentes = _avisos(r, "incoherente")
    assert any(a["feature"] == "weight" for a in incoherentes)


def test_el_aviso_no_toca_la_probabilidad(client, app_module, monkeypatch):
    """Mismo invariante que la capa clinica y AUD-16: informar sin alterar el numero.
    El MISMO payload incoherente, con el chequeo activo y con el chequeo apagado,
    tiene que dar exactamente la misma probabilidad (calibrada y cruda)."""
    incoherente = dict(bmi=19, waist_circumference=115, weight=55)
    con_aviso = _predict(client, **incoherente)
    assert _avisos(con_aviso, "incoherente")

    monkeypatch.setattr(app_module, "_check_coherencia_corporal", lambda *a: [])
    sin_aviso = _predict(client, **incoherente)
    assert _avisos(sin_aviso, "incoherente") == []

    assert con_aviso["probability"] == sin_aviso["probability"]
    assert con_aviso["raw_model_probability"] == sin_aviso["raw_model_probability"]


def test_campo_ausente_no_dispara_el_check(client, perfil_diabetes):
    """Sin weight/bmi/waist (otra enfermedad) el check ni se evalua."""
    r = client.post("/predict/diabetes", json=perfil_diabetes).get_json()
    assert _avisos(r, "incoherente") == []


def test_solo_aplica_a_hipertension(client):
    """cardiovascular tiene 'bmi' pero no weight/waist_circumference: no debe reventar
    ni generar avisos de coherencia corporal."""
    cardio = {
        "age": 50, "bmi": 45, "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1, "gender_Female": 1, "gender_Male": 0,
    }
    r = client.post("/predict/cardiovascular", json=cardio).get_json()
    assert _avisos(r, "incoherente") == []
