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
