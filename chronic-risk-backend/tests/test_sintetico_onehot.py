"""AUD-13 — el sintetico del GAN debe respetar los one-hot.

Antes, CTGAN veia `gender_Male` y `gender_Female` como dos binarias independientes y
las sampleaba por separado: en diabetes salian 1563 filas SIN genero y 729 con los
DOS a la vez, y solo el 45% del tabaquismo tenia exactamente una categoria. Se veia a
simple vista en la ficha de paciente del laboratorio y hacia trivial el juego
"real o sintetico". El fix colapsa cada grupo a una sola columna categorica antes de
entrenar y la expande al muestrear.

Estos tests cubren las dos mitades: los helpers (round-trip exacto) y el ARTEFACTO
que el API sirve de verdad (los CSV de data_curated).
"""
import glob
import math
import os

import pandas as pd
import pytest

import curate_and_synthesize as cs

ENFERMEDADES = ["diabetes", "hipertension", "cardiovascular"]


def _train(enf):
    return pd.read_csv(f"data_curated/{enf}/{enf}_train.csv")


def _sinteticos(enf):
    return sorted(glob.glob(f"data_curated/{enf}/{enf}_synthetic*.csv"))


# ---------- helpers de colapso/expansion ----------
@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_detecta_los_grupos_onehot(enf):
    grupos = cs._detect_onehot_groups(_train(enf))
    assert "gender" in grupos
    assert set(grupos["gender"]) == {"gender_Male", "gender_Female"}
    if enf != "cardiovascular":  # cardio (Kaggle) no trae tabaquismo categorizado
        assert len(grupos["smoking_history"]) == 3


@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_round_trip_exacto(enf):
    """Colapsar y expandir tiene que devolver EXACTAMENTE el dato original, incluidas
    las pocas filas sin categoria (7 en diabetes, 9 en hipertension)."""
    df = _train(enf)
    grupos = cs._detect_onehot_groups(df)
    ida = cs._colapsar_onehot(df, grupos)
    vuelta = cs._expandir_onehot(ida, grupos)[df.columns]
    pd.testing.assert_frame_equal(vuelta, df.astype(vuelta.dtypes.to_dict()))


def test_no_agrupa_columnas_que_no_son_onehot():
    """Dos binarias que pueden estar activas a la vez NO son un one-hot: agruparlas
    perderia informacion."""
    df = pd.DataFrame({
        "flag_a": [1, 1, 0, 0],
        "flag_b": [1, 0, 1, 0],   # coincide con flag_a en la primera fila
        "target": [0, 1, 0, 1],
    })
    assert cs._detect_onehot_groups(df) == {}


def test_columna_suelta_no_es_grupo():
    df = pd.DataFrame({"heart_disease": [0, 1], "age": [40, 50], "target": [0, 1]})
    assert cs._detect_onehot_groups(df) == {}


def test_fila_sin_categoria_recibe_su_propia_etiqueta():
    df = pd.DataFrame({
        "gender_Male": [1, 0, 0],
        "gender_Female": [0, 1, 0],   # la tercera fila no tiene genero
        "target": [0, 1, 0],
    })
    grupos = cs._detect_onehot_groups(df)
    col = cs._colapsar_onehot(df, grupos)
    assert list(col["gender"]) == ["gender_Male", "gender_Female", cs.ONEHOT_SIN_CATEGORIA]
    # y se expande de vuelta a "ninguna activa", sin inventarse un genero
    vuelta = cs._expandir_onehot(col, grupos)
    assert vuelta.loc[2, ["gender_Male", "gender_Female"]].tolist() == [0, 0]


def test_descarta_las_pocas_filas_sin_categoria():
    """Dato faltante disfrazado: si son pocas, fuera del entrenamiento del GAN (si no,
    CTGAN sobre-representa esa 'categoria' ~10x)."""
    df = pd.DataFrame({
        "gender_Male": [1, 0] * 50 + [0],
        "gender_Female": [0, 1] * 50 + [0],
        "target": [0, 1] * 50 + [0],
    })
    grupos = cs._detect_onehot_groups(df)
    assert len(cs._descartar_sin_categoria(df, grupos)) == 100


def test_si_son_muchas_no_se_descartan():
    """Con un hueco grande seria una categoria real del dominio: borrarla mentiria."""
    df = pd.DataFrame({
        "gender_Male": [1, 0, 0, 0],
        "gender_Female": [0, 1, 0, 0],
        "target": [0, 1, 0, 1],
    })
    grupos = cs._detect_onehot_groups(df)
    assert len(cs._descartar_sin_categoria(df, grupos)) == 4


# ---------- cuanto entrena el GAN (AUD-24) ----------
def test_las_epocas_se_derivan_del_tamano_del_dataset():
    """CTGAN cuenta pasos, no epocas: con --epochs fijo un dataset pequeno recibia
    10x menos entrenamiento que uno grande (correlaciones destruidas en el sintetico).
    Las epocas se derivan para que todos reciban los mismos pasos."""
    pequeno = cs._epocas_para(4789, 15000)     # hipertension
    grande = cs._epocas_para(54392, 15000)     # cardiovascular
    assert pequeno > grande * 5, (pequeno, grande)
    # y en pasos reales acaban en el mismo orden de magnitud
    pasos = lambda n, e: e * math.ceil(n / cs.BATCH_SIZE_CTGAN)
    assert 0.8 < pasos(4789, pequeno) / pasos(54392, grande) < 1.25


def test_nunca_menos_de_una_epoca():
    assert cs._epocas_para(10_000_000, 10) == 1


# ---------- el artefacto que sirve el API ----------
@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_los_csv_sinteticos_respetan_los_onehot(enf):
    """El test que importa: los CSV versionados, que son los que ve el usuario."""
    grupos = cs._detect_onehot_groups(_train(enf))
    ficheros = _sinteticos(enf)
    assert ficheros, f"no hay sintetico para {enf}"
    for f in ficheros:
        s = pd.read_csv(f)
        for prefijo, cols in grupos.items():
            suma = s[cols].sum(axis=1)
            assert (suma == 1).all(), (
                f"{os.path.basename(f)}: {(suma > 1).sum()} filas con varias "
                f"categorias de {prefijo} y {(suma == 0).sum()} sin ninguna")


@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_el_sintetico_conserva_el_esquema_del_real(enf):
    """Mismas columnas y en el mismo orden: la ficha de paciente y la comparacion de
    distribuciones asumen el esquema del train."""
    real = _train(enf)
    for f in _sinteticos(enf):
        assert list(pd.read_csv(f).columns) == list(real.columns), os.path.basename(f)


# ---------- lo que se ve por HTTP ----------
@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_las_fichas_sinteticas_tienen_un_solo_genero(client, enf):
    """El sintoma visible de AUD-13: pacientes sin genero o con los dos."""
    for _ in range(25):
        r = client.get(f"/synthetic/{enf}")
        assert r.status_code == 200
        fila = r.get_json()
        activos = [k for k in ("gender_Male", "gender_Female") if fila.get(k) == 1]
        assert len(activos) == 1, fila


def test_el_juego_real_vs_sintetico_no_se_delata(client):
    """Si el sintetico rompe el esquema, adivinar es trivial: basta mirar el genero."""
    for _ in range(25):
        fila = client.get("/sample/diabetes?source=synthetic").get_json()
        fumador = [k for k in fila if k.startswith("smoking_history_") and fila[k] == 1]
        genero = [k for k in fila if k.startswith("gender_") and fila[k] == 1]
        assert len(genero) == 1 and len(fumador) == 1, fila


# ---------- AUD-24: marginales categoricas ajustadas al real ----------
# CTGAN reequilibra las categorias durante el ajuste y el sintetico se desviaba entre
# 0.05 y 0.18 CON CUALQUIER numero de pasos (el colesterol alto de hipertension salia
# al 46-58% frente al 39.7% real). Se corrige generando de mas y submuestreando.
def _peor_marginal(real, synth, grupos):
    columnas = cs._columnas_categoricas(real, grupos) + [c for v in grupos.values() for c in v]
    return max(abs(float(real[c].mean()) - float(synth[c].mean()))
               for c in columnas if c in synth.columns)


@pytest.mark.parametrize("enf", ENFERMEDADES)
def test_las_marginales_categoricas_cuadran_con_el_real(enf):
    """El invariante que importa: lo que se sirve como 'población sintética' tiene la
    misma composición que la real."""
    real = _train(enf)
    grupos = cs._detect_onehot_groups(real)
    for f in _sinteticos(enf):
        peor = _peor_marginal(real, pd.read_csv(f), grupos)
        assert peor < 0.02, f"{os.path.basename(f)}: desviación máxima {peor:.3f}"


def test_el_reparto_de_cupos_suma_exacto_y_no_pierde_estratos():
    """Con un redondeo normal los estratos pequeños caían a cero y su masa la
    absorbían los grandes ronda tras ronda: eso sesgaba el resultado más que el
    problema que venía a arreglar. El reparto es por resto mayor."""
    props = pd.Series({"a": 0.90, "b": 0.06, "c": 0.03, "d": 0.01})
    cupos = cs._reparto_por_restos(props, 100)
    assert int(cupos.sum()) == 100
    assert (cupos > 0).all(), cupos.to_dict()   # ningun estrato se queda sin cupo


def test_el_ajuste_solo_elige_filas_del_pool():
    """No inventa nada: submuestrea. Cada fila devuelta tiene que existir en el pool."""
    real = _train("hipertension")
    pool = pd.read_csv(_sinteticos("hipertension")[0])
    grupos = cs._detect_onehot_groups(real)
    elegidas = cs._ajustar_marginales(real, pool, grupos, len(pool) // 4, 42)
    assert len(elegidas) == len(pool) // 4
    comunes = pd.merge(elegidas.round(6), pool.round(6), how="inner")
    assert len(comunes) >= len(elegidas)


def test_el_ajuste_mejora_la_desviacion():
    """La prueba de que el mecanismo hace lo que dice, sobre el artefacto real."""
    real = _train("hipertension")
    pool = pd.read_csv(_sinteticos("hipertension")[0])
    grupos = cs._detect_onehot_groups(real)
    # Se desequilibra el pool a proposito y se comprueba que el ajuste lo recompone.
    # El objetivo deja holgura: si se pidieran tantas filas como tiene el pool, no
    # habria de donde sacar las mujeres que faltan y el sesgo seria irreparable.
    sesgado = pd.concat([pool, pool[pool["gender_Male"] == 1]], ignore_index=True)
    antes = _peor_marginal(real, sesgado, grupos)
    ajustado = cs._ajustar_marginales(real, sesgado, grupos, len(pool) // 3, 42)
    despues = _peor_marginal(real, ajustado, grupos)
    assert antes > 0.1, antes            # el sesgo introducido es grande
    assert despues < 0.02, (antes, despues)
