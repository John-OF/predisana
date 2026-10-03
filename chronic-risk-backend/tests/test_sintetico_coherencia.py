"""Revision 2026-10 — el sintetico no puede traer pacientes imposibles.

CTGAN reproduce bien cada columna y mal las relaciones fuertes entre dos. En el
sintetico de hipertension la correlacion peso~IMC salia a 0,64 (real 0,89) y de ahi
100 pacientes con una talla implicita de 0,94 a 2,59 m y 96 con menos de 75 kg y mas
de 115 cm de cintura (en el real, 0 y 1); en cardiovascular, 150 con la sistolica
igual o por debajo de la diastolica (real, 0). Y dos pistas delataban al sintetico en
el juego "¿real o sintetico?": el peso con dos decimales (el real lleva uno) y el
pico de edad 80 de NHANES alisado (5% real, 1,6%-2,3% sintetico).

El generador ahora:
  - no le pide al GAN las relaciones fuertes: modela la talla y deriva el peso, y
    modela el residuo de cintura sobre IMC;
  - redondea cada columna a los decimales que usa el dato real;
  - descarta del pool las filas que el propio simulador marcaria como incoherentes;
  - trata "estar en el tope de edad" como un estrato mas al elegir filas del pool.

Como en test_sintetico_onehot.py, se cubren los helpers y el ARTEFACTO que sirve el API.
"""
import glob

import numpy as np
import pandas as pd
import pytest

import coherence
import curate_and_synthesize as cs


def _train(enf):
    return pd.read_csv(f"data_curated/{enf}/{enf}_train.csv")


def _sintetico(enf):
    return pd.read_csv(sorted(glob.glob(f"data_curated/{enf}/{enf}_synthetic*.csv"))[0])


# ---------- reparametrizacion ----------

def test_el_gan_no_ve_las_columnas_que_se_derivan():
    real = cs._sanitize_for_sdv(_train("hipertension"))
    gan, plan = cs._al_espacio_del_gan(real)
    assert {"weight", "waist_circumference"}.isdisjoint(gan.columns)
    assert len(plan) == 2
    assert len(gan.columns) == len(real.columns)   # una columna por cada una que quita


def test_la_reparametrizacion_es_reversible():
    real = cs._sanitize_for_sdv(_train("hipertension"))
    gan, plan = cs._al_espacio_del_gan(real)
    vuelta = cs._del_espacio_del_gan(gan, plan)[real.columns]
    assert float((vuelta - real).abs().max().max()) < 1e-9


@pytest.mark.parametrize("enf", ["cardiovascular", "diabetes"])
def test_no_se_reparametriza(enf):
    """Cardiovascular: su presion va en multiplos de 10 y un residuo continuo destruiria
    ese patron. Diabetes: el residuo de glucosa sobre HbA1c se probo, lineal y en log, y
    en los dos empeoraba la relacion de la glucosa con el target; el lineal, ademas,
    dejaba el 1,7% del sintetico clavado en 40 mg/dL."""
    real = cs._sanitize_for_sdv(_train(enf))
    gan, plan = cs._al_espacio_del_gan(real)
    assert plan == [] and list(gan.columns) == list(real.columns)


def test_la_talla_casi_no_depende_del_imc():
    """Por eso se le da al GAN la talla y no el peso: peso e IMC van juntos (r=0,89),
    talla e IMC no. Lo que el GAN no aprende de una relacion que no existe no rompe nada."""
    gan, _ = cs._al_espacio_del_gan(cs._sanitize_for_sdv(_train("hipertension")))
    real = _train("hipertension")
    assert real["weight"].corr(real["bmi"]) > 0.85
    assert abs(gan[cs.COLUMNA_TALLA].corr(gan["bmi"])) < 0.1


def test_lo_derivado_se_topa_al_rango_real():
    """IMC y talla maximos a la vez darian un peso que nadie tiene: se topa, igual que
    SDV topa las columnas que si modela."""
    real = cs._sanitize_for_sdv(_train("hipertension"))
    gan, plan = cs._al_espacio_del_gan(real)
    extremo = gan.iloc[[0]].copy()
    extremo["bmi"], extremo[cs.COLUMNA_TALLA] = gan["bmi"].max(), gan[cs.COLUMNA_TALLA].max()
    vuelta = cs._del_espacio_del_gan(extremo, plan)
    assert vuelta["weight"].iloc[0] == real["weight"].max()


# ---------- precision del dato real ----------

def test_decimales_cuenta_lo_que_usa_la_columna_no_su_peor_fila():
    """Una fila real de 4798 trae 62.87 y SDV, que toma el maximo, sacaba el peso
    sintetico con dos decimales."""
    assert cs._decimales(pd.Series([62.2, 70.5, 81.0] * 100 + [62.87])) == 1
    assert cs._decimales(pd.Series([25.0, 80.0, 41.0])) == 0
    assert cs._decimales(pd.Series(np.sqrt(np.arange(2, 60)))) == cs.MAX_DECIMALES + 1
    assert cs._decimales(_train("hipertension")["weight"]) == 1


def test_redondea_cada_columna_como_el_real():
    real = pd.DataFrame({"peso": [62.2, 70.5], "edad": [30.0, 41.0],
                         "imc": [21.96712, 30.11234], "target": [0, 1]})
    muestra = pd.DataFrame({"peso": [64.4567], "edad": [33.6], "imc": [25.123456789], "target": [1]})
    out = cs._redondear_como_el_real(muestra, real)
    assert out.loc[0, "peso"] == 64.5
    assert out.loc[0, "edad"] == 34.0
    assert out.loc[0, "imc"] == 25.123456789      # continua de verdad: no se toca


# ---------- filas imposibles ----------

def test_descarta_del_pool_lo_que_el_simulador_marcaria():
    pool = pd.DataFrame({"ap_hi": [120, 100, 110], "ap_lo": [80, 120, 110], "age": [50, 51, 52]})
    assert cs._descartar_incoherentes("cardiovascular", pool)["age"].tolist() == [50]
    pool = pd.DataFrame({"weight": [75, 55, 180], "bmi": [25, 19, 25],
                         "waist_circumference": [85, 115, 85]})
    assert len(cs._descartar_incoherentes("hipertension", pool)) == 1


def test_sin_reglas_no_descarta_nada():
    pool = pd.DataFrame({"age": [30, 40], "bmi": [20, 50]})
    assert len(cs._descartar_incoherentes("diabetes", pool)) == 2


# ---------- tope de edad ----------

@pytest.mark.parametrize("enf,esperado", [
    ("diabetes", {"age": 80.0}), ("hipertension", {"age": 80.0}), ("cardiovascular", {}),
])
def test_detecta_el_tope_de_edad_de_nhanes(enf, esperado):
    """Y no confunde con un tope el valor 3 de una escala 1-3 (colesterol)."""
    real = _train(enf)
    assert cs._columnas_con_tope(real, cs._detect_onehot_groups(real)) == esperado


def test_el_tope_es_un_estrato_al_elegir_filas():
    """Un pool sin nadie de 80 no puede dar un sintetico con el 5% de 80: se queda sin
    ellos. Uno que los tiene de sobra devuelve la proporcion real."""
    real = _train("hipertension")
    grupos = cs._detect_onehot_groups(real)
    pool = pd.concat([_sintetico("hipertension")] * 3, ignore_index=True)
    viejos = pool[pool["age"] >= 70].copy()
    viejos["age"] = 80.0
    elegido = cs._ajustar_marginales(real, pd.concat([pool, viejos], ignore_index=True),
                                     grupos, len(real) // 2, 42)
    assert abs((elegido["age"] == 80).mean() - (real["age"] == 80).mean()) < 0.01


# ---------- el artefacto que sirve el API ----------

@pytest.mark.parametrize("enf", ["hipertension", "cardiovascular"])
def test_ningun_caso_sintetico_es_incoherente(enf):
    """Lo que se pide: un "caso virtual" cargado en el simulador no dispara el aviso de
    incoherencia. Antes lo disparaban 166 filas de hipertension y 150 de cardiovascular."""
    campos = list(coherence.CAMPOS[enf])
    malas = _sintetico(enf)[campos].apply(
        lambda fila: bool(coherence.incoherencias(enf, fila.to_dict())), axis=1)
    assert int(malas.sum()) == 0


@pytest.mark.parametrize("enf", ["diabetes", "hipertension", "cardiovascular"])
def test_el_sintetico_usa_los_decimales_del_real(enf):
    real, sint = _train(enf), _sintetico(enf)
    for c in real.columns:
        d = cs._decimales(real[c])
        if c != "target" and d <= cs.MAX_DECIMALES:
            assert cs._decimales(sint[c]) <= d, c


@pytest.mark.parametrize("enf", ["diabetes", "hipertension"])
def test_el_pico_de_edad_80_no_se_alisa(enf):
    real, sint = _train(enf), _sintetico(enf)
    assert abs((sint["age"] == 80).mean() - (real["age"] == 80).mean()) < 0.015


def test_la_glucosa_no_se_amontona_en_el_minimo():
    """Con el residuo lineal de glucosa sobre HbA1c, el 1,7% del sintetico quedaba
    clavado en 40 mg/dL (real, 0%)."""
    real, sint = _train("diabetes"), _sintetico("diabetes")
    assert (sint["blood_glucose_level"] == real["blood_glucose_level"].min()).mean() < 0.005


def test_el_sintetico_no_sale_del_rango_del_real():
    """Las columnas derivadas se topan al rango real, como las que modela SDV."""
    for enf in ("diabetes", "hipertension"):
        real, sint = _train(enf), _sintetico(enf)
        for c in ("weight", "waist_circumference", "blood_glucose_level"):
            if c in real.columns:
                assert real[c].min() <= sint[c].min() and sint[c].max() <= real[c].max(), (enf, c)
