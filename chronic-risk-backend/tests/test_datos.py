"""Datos v2 (prepare_nhanes.py): NHANES 2017-2020 + 2021-2023, enfermedad total.

Lo que la auditoria de datos de la revision 2026-10 encontro y estos tests fijan:
- el objetivo es la enfermedad total, asi que nadie con HbA1c >= 6,5, ayunas >= 126,
  presion >= 140/90, filtrado < 60 o albumina en orina >= 30 puede figurar como sano;
- ninguna variable puede significar cosas distintas segun el ciclo: la actividad
  fisica se quito porque el cuestionario cambio (34% frente a 53% cumplian los 150
  min de la OMS), y este test la habria parado;
- el IMC autodeclarado no es el medido (la gente se lo baja), por eso el modo
  simplificado se entrena con el autodeclarado.
"""
import numpy as np
import pandas as pd
import pytest

import modos as M

ENFERMEDADES = M.ENFERMEDADES
CICLOS = {"2017-2020", "2021-2023"}
NO_VARIABLES = ("ciclo", "peso_entrevista", "peso_examen", "target")


@pytest.fixture(scope="module", params=ENFERMEDADES)
def datos(request):
    return request.param, pd.read_csv(f"data_processed/{request.param}_dataset.csv")


def _binarias(d):
    return [c for c in d.columns if c not in NO_VARIABLES[:3]
            and set(d[c].dropna().unique()) <= {0, 1}]


def test_tiene_los_dos_ciclos_y_mas_de_diez_mil_adultos(datos):
    _, d = datos
    assert set(d["ciclo"]) == CICLOS
    assert len(d) > 10_000
    assert set(d["target"]) == {0, 1}


def test_cada_fila_tiene_el_peso_muestral_que_le_toca(datos):
    """El peso del examen es 0 para quien solo hizo la entrevista: en cardiovascular,
    cuyo objetivo sale de la entrevista, son 2.412 personas."""
    _, d = datos
    assert (d["peso_entrevista"] > 0).all()
    examinados = d["bmi"].notna()
    assert (d.loc[examinados, "peso_examen"] > 0).all()


def test_sin_codigos_de_nhanes_en_las_binarias(datos):
    """7 (se niega) y 9 (no sabe) son dato faltante, no un "no" ni una categoria."""
    _, d = datos
    for c in _binarias(d):
        assert d[c].dropna().isin([0, 1]).all(), c
    assert len(_binarias(d)) >= 8


def test_los_one_hot_suman_uno(datos):
    _, d = datos
    assert (d["gender_Male"] + d["gender_Female"] == 1).all()
    tabaco = d[["smoking_history_never", "smoking_history_current", "smoking_history_former"]]
    conocido = tabaco.notna().all(axis=1)
    assert (tabaco[conocido].sum(axis=1) == 1).all()
    assert tabaco[~conocido].isna().all(axis=1).all()


def test_ninguna_variable_cambia_de_significado_entre_ciclos(datos):
    nombre, d = datos
    a, b = d[d["ciclo"] == "2017-2020"], d[d["ciclo"] == "2021-2023"]
    for c in d.columns.drop(list(NO_VARIABLES)):
        x, y = a[c].dropna(), b[c].dropna()
        if c in _binarias(d):
            assert abs(x.mean() - y.mean()) < 0.08, (nombre, c, x.mean(), y.mean())
        else:
            d_cohen = abs(x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)
            assert d_cohen < 0.2, (nombre, c, d_cohen)


def test_la_prevalencia_crece_con_la_edad(datos):
    nombre, d = datos
    tramos = [d[d["age"].between(lo, lo + 19)]["target"].mean() for lo in (20, 40, 60)]
    if nombre == "higado":
        # La grasa en el higado sube hasta los 40 y se queda en una meseta (39% y 38%).
        assert tramos[0] < tramos[1] and tramos[2] > tramos[0] + 0.1, (nombre, tramos)
        return
    assert tramos == sorted(tramos), (nombre, tramos)


def test_el_imc_autodeclarado_se_queda_corto_con_obesidad(datos):
    _, d = datos
    obesos = d["bmi"] >= 35
    assert (d["bmi_autodeclarado"] - d["bmi"])[obesos].mean() < -1


# ---------- el objetivo es la enfermedad total ----------

def _uno(nombre):
    return pd.read_csv(f"data_processed/{nombre}_dataset.csv")


def test_diabetes_por_analitica_cuenta_aunque_no_este_diagnosticada():
    d = _uno("diabetes")
    por_lab = (d["hba1c_level"] >= 6.5) | (d["blood_glucose_level"] >= 126)
    assert (d.loc[por_lab, "target"] == 1).all()
    assert d["hba1c_level"].notna().all()          # sin HbA1c no se puede saber
    assert "diabetes" not in d.columns              # el diagnostico es parte del objetivo


def test_hipertension_medida_cuenta_aunque_no_este_diagnosticada():
    d = _uno("hipertension")
    alta = (d["ap_hi"] >= 140) | (d["ap_lo"] >= 90)
    assert (d.loc[alta, "target"] == 1).all()
    assert d["ap_hi"].notna().all()
    assert "hypertension" not in d.columns


def test_cardiovascular_empieza_a_los_20_y_las_cardiopatias_de_18_19_son_cero():
    """MCQ160 solo se pregunta desde los 20: a los de 18-19 les falta por diseno."""
    assert _uno("cardiovascular")["age"].min() >= 20
    for nombre in ("diabetes", "hipertension", "renal"):
        jovenes = _uno(nombre).query("age < 20")
        assert len(jovenes) > 100 and (jovenes["heart_disease"] == 0).all()


def test_higado_es_cap_288_en_una_exploracion_valida():
    """Esteatosis por elastografia (CAP >= 288 dB/m); solo entra quien tiene una
    exploracion valida. No hay diagnostico autodeclarado: es una medicion."""
    d = _uno("higado")
    assert len(d) > 12000
    assert d["target"].mean() == pytest.approx(0.335, abs=0.01)
    assert d["alt"].dropna().between(1, 700).all()
    # sube con el IMC y con la ALT
    por_imc = d.groupby(pd.cut(d["bmi"], [0, 25, 30, 35, 100]), observed=True)["target"].mean()
    assert list(por_imc) == sorted(por_imc)
    por_alt = d.groupby(pd.qcut(d["alt"], 4), observed=True)["target"].mean()
    assert list(por_alt) == sorted(por_alt)


def _renal_por_analitica(d):
    return (d["egfr"] < 60) | (d["albumin_creatinine_ratio"] >= 30)


def test_renal_por_analitica_cuenta_aunque_no_este_diagnosticada():
    """KDIGO: filtrado < 60 o albumina/creatinina >= 30 mg/g. Solo el 19% de quienes
    la tienen estaba diagnosticado; el diagnostico sin analitica alterada es el 1%."""
    d = _uno("renal")
    por_lab = _renal_por_analitica(d)
    assert (d.loc[por_lab, "target"] == 1).all()
    assert d[["egfr", "albumin_creatinine_ratio"]].notna().all().all()   # sin las dos no se sabe
    assert 0 < d.loc[~por_lab, "target"].mean() < 0.03
    assert d["target"].mean() == pytest.approx(0.188, abs=0.005)


def test_renal_de_18_19_la_define_solo_la_analitica():
    """KIQ022 ("rinones debiles o en fallo") se pregunta desde los 20."""
    d = _uno("renal")
    jovenes = d[d["age"] < 20]
    assert len(jovenes) > 100
    assert (jovenes["target"] == _renal_por_analitica(jovenes).astype(int)).all()
