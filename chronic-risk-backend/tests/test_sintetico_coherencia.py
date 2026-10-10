"""Revision 2026-10 — el sintetico no puede traer pacientes imposibles.

CTGAN reproduce bien cada columna y mal las relaciones fuertes entre dos. En el
sintetico v1 de hipertension la correlacion peso~IMC salia a 0,64 (real 0,89) y de ahi
100 pacientes con una talla implicita de 0,94 a 2,59 m y 96 con menos de 75 kg y mas
de 115 cm de cintura (en el real, 0 y 1); en cardiovascular, 150 con la sistolica
igual o por debajo de la diastolica (real, 0). Y dos pistas delataban al sintetico en
el juego "¿real o sintetico?": el peso con dos decimales (el real lleva uno) y el
pico de edad 80 de NHANES alisado (5% real, 1,6%-2,3% sintetico).

El generador ahora:
  - no le pide al GAN las relaciones fuertes: deriva el peso del IMC declarado y la
    talla, y modela el residuo de la cintura y del IMC declarado sobre el IMC medido y
    el de la diastolica sobre la sistolica;
  - redondea cada columna a los decimales que usa el dato real;
  - descarta del pool las filas que el propio simulador marcaria como incoherentes;
  - trata "estar en el tope de edad" como un estrato mas al elegir filas del pool.

Desde la v2 el GAN ve las columnas del laboratorio (modos.columnas_laboratorio) de las
filas completas del train: ese es el real con el que se compara. Como en
test_sintetico_onehot.py, se cubren los helpers y el ARTEFACTO que sirve el API.
"""
import glob

import numpy as np
import pandas as pd
import pytest

import coherence
import curate_and_synthesize as cs
import modos as M

DERIVADAS = ("weight", "waist_circumference", "bmi_autodeclarado", "ap_lo")


def _real(enf):
    """Lo que ve el GAN: las columnas del laboratorio de las filas completas del train."""
    tr = pd.read_csv(f"data_curated/{enf}/{enf}_train.csv")
    return tr[M.columnas_laboratorio(enf)].dropna().reset_index(drop=True)


def _sintetico(enf):
    return pd.read_csv(sorted(glob.glob(f"data_curated/{enf}/{enf}_synthetic*.csv"))[0])


# ---------- reparametrizacion ----------

@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_el_gan_no_ve_las_columnas_que_se_derivan(enf):
    real = cs._sanitize_for_sdv(_real(enf))
    gan, plan = cs._al_espacio_del_gan(real)
    assert set(DERIVADAS).isdisjoint(gan.columns)
    assert [p.get("dep", p["tipo"]) for p in plan] == [
        "peso", "waist_circumference", "bmi_autodeclarado", "ap_lo"]
    # Cada residuo ocupa el lugar de su columna; el peso sale de dos que el GAN si ve.
    assert len(gan.columns) == len(real.columns) - 1


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_la_reparametrizacion_es_reversible(enf):
    """Exacta salvo el peso, que vuelve como IMC declarado x talla^2: ese IMC se guarda
    con un decimal, asi que el peso se desvia hasta un cuarto de kilo."""
    real = cs._sanitize_for_sdv(_real(enf))
    gan, plan = cs._al_espacio_del_gan(real)
    error = (cs._del_espacio_del_gan(gan, plan)[real.columns] - real).abs().max()
    assert error.drop("weight").max() < 1e-9
    assert error["weight"] < 0.3


def test_el_peso_va_con_el_imc_y_la_talla_no():
    """Por eso el GAN ve la talla y no el peso: lo que no aprende de una relacion que
    no existe no rompe nada."""
    real = _real("hipertension")
    assert real["weight"].corr(real["bmi_autodeclarado"]) > 0.85
    assert abs(real["height"].corr(real["bmi_autodeclarado"])) < 0.1


def test_lo_derivado_se_topa_al_rango_real():
    """IMC y talla maximos a la vez darian un peso que nadie tiene: se topa, igual que
    SDV topa las columnas que si modela."""
    real = cs._sanitize_for_sdv(_real("hipertension"))
    gan, plan = cs._al_espacio_del_gan(real)
    extremo = gan.iloc[[0]].copy()
    for c in ("bmi", "height", cs.PREFIJO_RESIDUO + "bmi_autodeclarado"):
        extremo[c] = gan[c].max()
    vuelta = cs._del_espacio_del_gan(extremo, plan)
    assert vuelta["bmi_autodeclarado"].iloc[0] == real["bmi_autodeclarado"].max()
    assert vuelta["weight"].iloc[0] == real["weight"].max()


# ---------- la presion, el objetivo y la obesidad (revision 2026-10) ----------

def test_el_residuo_no_depende_de_su_base_y_tiene_la_escala_real():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(120, 18, 5000))
    residuo = pd.Series(-0.2 * (base - 120) + rng.normal(0, 12, 5000))   # el GAN se inventa la relacion
    limpio = cs._a_la_escala_real(cs._sin_correlacion_con(residuo, base), 8.7)
    assert abs(limpio.corr(base)) < 1e-9
    assert limpio.std() == pytest.approx(8.7) and limpio.mean() == pytest.approx(0, abs=1e-9)


def test_una_sola_fila_no_se_toca():
    uno = pd.Series([3.0])
    assert cs._a_la_escala_real(cs._sin_correlacion_con(uno, pd.Series([5.0])), 8.7).iloc[0] == 3.0


@pytest.mark.parametrize("enf", sorted(cs.REGLAS_OBJETIVO))
def test_quien_cumple_la_regla_de_laboratorio_tiene_la_enfermedad(enf):
    """En el real es exacto (HbA1c >= 6,5; 140/90; filtrado < 60 o albumina >= 30). El
    GAN lo daba al 67%, 69% y 41%: fichas con la HbA1c en 8 y 'sin diabetes'."""
    sint = _sintetico(enf)
    cumple = cs.REGLAS_OBJETIVO[enf](sint)
    assert cumple.mean() > 0.1
    assert (sint.loc[cumple, "target"] == 1).all()


@pytest.mark.parametrize("enf", sorted(cs.REGLAS_OBJETIVO))
def test_la_definicion_del_objetivo_deja_la_tasa_de_los_diagnosticados_sin_regla_al_real(enf):
    """Quien tiene la enfermedad sin cumplir la regla (diagnosticado, tratado) lo daba
    el GAN de mas: renal 16% frente al 1,2% real."""
    real, sint = _real(enf), _sintetico(enf)
    q_real = real.loc[~cs.REGLAS_OBJETIVO[enf](real), "target"].mean()
    q_sint = sint.loc[~cs.REGLAS_OBJETIVO[enf](sint), "target"].mean()
    assert abs(q_sint - q_real) < 0.06, (q_real, q_sint)


def test_aplicar_la_definicion_solo_cambia_el_target():
    real = _real("renal")
    pool = _sintetico("renal").assign(target=0)
    fuera = cs._aplicar_definicion_del_objetivo(pool, real, "renal", seed=1)
    assert fuera.drop(columns="target").equals(pool.drop(columns="target"))
    assert (fuera["target"] == cs.REGLAS_OBJETIVO["renal"](pool).astype(int)).all()
    # sin regla (cardiovascular, higado) no hace nada
    assert cs._aplicar_definicion_del_objetivo(pool, real, "cardiovascular", 1) is pool


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_la_obesidad_grave_no_se_sobrerrepresenta(enf):
    """Cardiovascular y renal sacaban un 30-33% de IMC >= 35 frente al 20% real."""
    real, sint = _real(enf), _sintetico(enf)
    assert abs((sint["bmi"] >= 35).mean() - (real["bmi"] >= 35).mean()) < 0.02


@pytest.mark.parametrize("enf", ["diabetes", "hipertension", "cardiovascular"])
def test_la_correlacion_de_la_presion_cuadra_con_el_real(enf):
    """Hipertension y cardiovascular salian a 0,43-0,45 con un real de 0,61."""
    real, sint = _real(enf), _sintetico(enf)
    assert abs(sint["ap_hi"].corr(sint["ap_lo"]) - real["ap_hi"].corr(real["ap_lo"])) < 0.05


# ---------- precision del dato real ----------

def test_decimales_cuenta_lo_que_usa_la_columna_no_su_peor_fila():
    """Una fila real de 4798 trae 62.87 y SDV, que toma el maximo, sacaba el peso
    sintetico con dos decimales."""
    assert cs._decimales(pd.Series([62.2, 70.5, 81.0] * 100 + [62.87])) == 1
    assert cs._decimales(pd.Series([25.0, 80.0, 41.0])) == 0
    assert cs._decimales(pd.Series(np.sqrt(np.arange(2, 60)))) == cs.MAX_DECIMALES + 1
    assert cs._decimales(_real("hipertension")["weight"]) == 1


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
    assert cs._descartar_incoherentes(pool)["age"].tolist() == [50]
    pool = pd.DataFrame({"bmi": [25, 19, 42], "waist_circumference": [85, 115, 70]})
    assert len(cs._descartar_incoherentes(pool)) == 1


def test_el_peso_declarado_no_se_cruza_con_el_imc_medido():
    """Igual que en el modo completo de la API: son dos mediciones distintas de la
    misma persona, y juntas no implican ninguna talla."""
    pool = pd.DataFrame({"weight": [40.0], "bmi": [30.0], "waist_circumference": [95.0]})
    assert len(cs._descartar_incoherentes(pool)) == 1


def test_sin_reglas_no_descarta_nada():
    pool = pd.DataFrame({"age": [30, 40], "bmi": [20, 50]})
    assert len(cs._descartar_incoherentes(pool)) == 2


# ---------- tope de edad ----------

@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_detecta_el_tope_de_edad_de_nhanes(enf):
    real = _real(enf)
    assert cs._columnas_con_tope(real, cs._detect_onehot_groups(real)) == {"age": 80.0}


def test_una_escala_corta_no_es_un_tope():
    """El valor 3 de una escala 1-3 (el colesterol de la v1 de cardiovascular) no es
    un tope de codificacion."""
    escala = pd.DataFrame({"escala": [1, 2, 3, 3] * 50, "target": [0, 1] * 100})
    assert cs._columnas_con_tope(escala, {}) == {}


def test_el_tope_es_un_estrato_al_elegir_filas():
    """Un pool sin nadie de 80 no puede dar un sintetico con el 5% de 80: se queda sin
    ellos. Uno que los tiene de sobra devuelve la proporcion real."""
    real = _real("hipertension")
    grupos = cs._detect_onehot_groups(real)
    pool = pd.concat([_sintetico("hipertension")] * 3, ignore_index=True)
    viejos = pool[pool["age"] >= 70].copy()
    viejos["age"] = 80.0
    elegido = cs._ajustar_marginales(real, pd.concat([pool, viejos], ignore_index=True),
                                     grupos, len(real) // 2, 42)
    assert abs((elegido["age"] == 80).mean() - (real["age"] == 80).mean()) < 0.01


# ---------- el artefacto que sirve el API ----------

@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_ningun_caso_sintetico_es_incoherente(enf):
    """Lo que se pide: un "caso virtual" que el laboratorio evalua no dispara el aviso
    de incoherencia. En la v1 lo disparaban 166 filas de hipertension y 150 de
    cardiovascular. El laboratorio evalua en el completo, que no juzga el peso."""
    campos = [c for c in coherence.CAMPOS if c != "weight"]
    malas = _sintetico(enf)[campos].apply(
        lambda fila: bool(coherence.incoherencias(fila.to_dict())), axis=1)
    assert int(malas.sum()) == 0


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_las_fichas_mas_extremas_pasan_por_la_api_sin_aviso(client, enf):
    """La misma comprobacion por donde pasa el usuario: las fichas con la presion mas
    apretada y con la cintura mas rara para su IMC, evaluadas como lo hace el
    laboratorio (la ficha entera, modo completo)."""
    s = _sintetico(enf)
    cintura_imc = s["waist_circumference"] / s["bmi"]
    extremas = [(s["ap_hi"] - s["ap_lo"]).nsmallest(15).index,
                cintura_imc.nsmallest(15).index, cintura_imc.nlargest(15).index]
    for _, ficha in pd.concat([s.loc[i] for i in extremas]).iterrows():
        r = client.post(f"/predict/{enf}?mode=completo&source=synthetic", json=ficha.to_dict())
        assert r.status_code == 200, r.get_json()
        assert not [a for a in r.get_json()["support_warnings"] if a["level"] == "incoherente"]


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_el_sintetico_usa_los_decimales_del_real(enf):
    real, sint = _real(enf), _sintetico(enf)
    for c in real.columns:
        d = cs._decimales(real[c])
        if c != "target" and d <= cs.MAX_DECIMALES:
            assert cs._decimales(sint[c]) <= d, c


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_el_pico_de_edad_80_no_se_alisa(enf):
    real, sint = _real(enf), _sintetico(enf)
    assert abs((sint["age"] == 80).mean() - (real["age"] == 80).mean()) < 0.015


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
@pytest.mark.parametrize("par", [("weight", "bmi_autodeclarado"), ("waist_circumference", "bmi"),
                                 ("bmi_autodeclarado", "bmi")])
def test_las_relaciones_fuertes_no_se_recortan(enf, par):
    """Lo que se gana con la reparametrizacion: sin ella el GAN de la v1 dejaba
    peso~IMC en 0,64 (real 0,89)."""
    real, sint = _real(enf), _sintetico(enf)
    a, b = par
    assert abs(sint[a].corr(sint[b]) - real[a].corr(real[b])) < 0.1


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_la_diastolica_sigue_a_la_sistolica(enf):
    """Sin el residuo, el GAN dejaba sistolica~diastolica en 0,30 (real 0,61). Con el
    residuo, diabetes sale 0,62, pero hipertension y cardiovascular se quedan en
    0,43-0,45: el GAN aprende una relacion entre el residuo y la sistolica (-0,15 y
    -0,20) que en el real es cero por construccion. Queda abierto; esto impide volver
    atras."""
    real, sint = _real(enf), _sintetico(enf)
    r = sint["ap_lo"].corr(sint["ap_hi"])
    assert 0.4 < r < real["ap_lo"].corr(real["ap_hi"]) + 0.1


@pytest.mark.parametrize("enf", M.ENFERMEDADES)
def test_el_sintetico_no_sale_del_rango_del_real(enf):
    """Las columnas derivadas se topan al rango real, como las que modela SDV."""
    real, sint = _real(enf), _sintetico(enf)
    for c in DERIVADAS:
        assert real[c].min() <= sint[c].min() and sint[c].max() <= real[c].max(), c
