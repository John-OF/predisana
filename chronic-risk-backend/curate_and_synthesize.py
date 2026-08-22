# curate_and_synthesize.py
import os
import math
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# --------- RUTAS ---------
PROCESSED_DIR = "data_processed"
CURATED_DIR = "data_curated"

DATASETS = ["diabetes", "hipertension", "cardiovascular"]
FILENAME = "{name}_dataset.csv"  # dentro de data_processed

# --------- IMPORTS CON FALLBACK (SDV) ---------
SDV_AVAILABLE = True
SDV_API_V1 = False
_SDVi_err = None
try:
    # SDV >= 1.x
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_API_V1 = True
except Exception as e1:
    try:
        # SDV 0.x
        from sdv.tabular import CTGAN, TVAE  # type: ignore
        SDV_API_V1 = False
    except Exception as e2:
        SDV_AVAILABLE = False
        _SDVi_err = (e1, e2)

# --------- UTILS ---------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _is_binary(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0: return False
    v = pd.to_numeric(vals, errors="coerce")
    if v.notna().all():
        v = v.astype(int).unique()
        return set(v).issubset({0, 1})
    return False

def stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    y = df["target"]
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=y)

def _detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    cats = []
    for c in df.columns:
        if c == "target":
            cats.append(c); continue
        if c.startswith("gender_") or c.startswith("smoking_history_"):
            cats.append(c); continue
        if _is_binary(df[c]):
            cats.append(c)
    return cats

# --------- CUANTO ENTRENAR EL GAN (AUD-24) ---------
# CTGAN no cuenta epocas, cuenta PASOS de gradiente: ceil(n_filas / batch_size) por
# epoca. Con --epochs fijo, un dataset de 4.8k filas recibia 10x menos entrenamiento
# que uno de 55k, y se notaba: a 50 epocas la correlacion peso-cintura del sintetico
# de hipertension era -0.08 (real +0.90) y el 22% de las filas eran fisicamente
# imposibles (peso <75 kg con cintura >115 cm). Por eso el parametro natural es el
# numero de pasos, y las epocas se derivan de el para cada dataset.
BATCH_SIZE_CTGAN = 500  # el default de SDV; si se cambia alli, cambiarlo aqui

# Mas entrenamiento no es gratis: el training-by-sampling de CTGAN aplana las
# marginales categoricas segun avanza. Medido en cardiovascular al pasar de 5,4k a
# 15k pasos: la proporcion de hombres se va del 0.350 real al 0.485 (hacia el 50/50),
# y colesterol y glucosa se desplazan igual; su score SDMetrics baja de 0.924 a 0.893
# aunque la correlacion ap_hi/ap_lo mejore (0.47 -> 0.65). Como ya venia bien
# entrenado (55k filas dan 109 pasos por epoca), se queda donde estaba. Los datasets
# pequenos SI mejoran en todo con 15k pasos, que es el default.
PASOS_POR_DATASET = {"cardiovascular": 5500}


def _epocas_para(n_filas: int, pasos_objetivo: int) -> int:
    pasos_por_epoca = max(1, math.ceil(n_filas / BATCH_SIZE_CTGAN))
    return max(1, round(pasos_objetivo / pasos_por_epoca))


# --------- ONE-HOT COMO UNA SOLA CATEGORICA (AUD-13) ---------
# El GAN veia `gender_Male` y `gender_Female` como dos binarias independientes, asi
# que sampleaba cada una por su cuenta: salian filas sin genero (0,0) y con los dos
# a la vez (1,1) — 1563 y 729 en el sintetico de diabetes, y solo el 45% del
# tabaquismo tenia exactamente una categoria. Se ve a simple vista en la ficha de
# paciente del laboratorio y hace trivial el juego "real o sintetico".
# La solucion no es limpiar despues (un argmax inventaria un genero donde el GAN no
# eligio ninguno): es colapsar cada grupo a UNA columna categorica antes de entrenar,
# para que el modelo aprenda que son mutuamente excluyentes, y expandirla al volver.
ONEHOT_SIN_CATEGORIA = "__sin_categoria__"


def _detect_onehot_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Agrupa columnas que comparten prefijo (`gender_*`, `smoking_history_*`) y que
    se comportan como un one-hot: todas binarias y como mucho UNA activa por fila.
    Se detecta sobre los datos reales, asi que un dataset nuevo no necesita tocar
    codigo. Si el grupo no cumple la exclusividad, se deja tal cual."""
    candidatos: Dict[str, List[str]] = {}
    for c in df.columns:
        if c == "target" or "_" not in c:
            continue
        candidatos.setdefault(c.rsplit("_", 1)[0], []).append(c)

    grupos: Dict[str, List[str]] = {}
    for prefijo, cols in candidatos.items():
        if len(cols) < 2 or prefijo in df.columns:
            continue
        if not all(_is_binary(df[c]) for c in cols):
            continue
        if (df[cols].fillna(0).sum(axis=1) > 1).any():
            continue
        grupos[prefijo] = sorted(cols)
    return grupos


def _colapsar_onehot(df: pd.DataFrame, grupos: Dict[str, List[str]]) -> pd.DataFrame:
    """gender_Male=1, gender_Female=0  ->  gender="gender_Male"."""
    out = df.copy()
    for prefijo, cols in grupos.items():
        valores = out[cols].astype(float)
        # Las filas sin ninguna activa existen de verdad (p.ej. 7 filas de diabetes
        # sin tabaquismo declarado): se les da su propia categoria en vez de
        # inventarles una, para que el round-trip sea exacto.
        etiqueta = valores.idxmax(axis=1).where(valores.sum(axis=1) > 0, ONEHOT_SIN_CATEGORIA)
        out = out.drop(columns=cols)
        out[prefijo] = etiqueta.astype(str)
    return out


def _expandir_onehot(df: pd.DataFrame, grupos: Dict[str, List[str]]) -> pd.DataFrame:
    """Inversa de `_colapsar_onehot`: deja exactamente una columna a 1 (o ninguna)."""
    out = df.copy()
    for prefijo, cols in grupos.items():
        if prefijo not in out.columns:
            continue
        etiqueta = out[prefijo].astype(str)
        out = out.drop(columns=[prefijo])
        for c in cols:
            out[c] = (etiqueta == c).astype("int64")
    return out


def _descartar_sin_categoria(df: pd.DataFrame, grupos: Dict[str, List[str]],
                            umbral: float = 0.02) -> pd.DataFrame:
    """Quita del entrenamiento del GAN las filas sin ninguna categoria activa.
    Son datos faltantes disfrazados (el "No Info" que se perdio al hacer el one-hot):
    7 filas de 4987 en diabetes, 9 de 4798 en hipertension. Si se dejan, CTGAN las
    trata como una categoria mas y su training-by-sampling la sobre-representa ~10x
    (86 filas sinteticas sin tabaquismo, de 9 reales). Es la misma politica que ya
    aplica `_sanitize_for_sdv` con los NaN: el GAN se entrena con casos completos.
    Si el hueco fuera grande (>umbral) NO se descarta: ahi seria una categoria real
    del dominio y borrarla mentiria sobre los datos."""
    if not grupos:
        return df
    incompletas = pd.Series(False, index=df.index)
    for cols in grupos.values():
        incompletas |= (df[cols].fillna(0).sum(axis=1) == 0)
    n = int(incompletas.sum())
    if n == 0:
        return df
    if n / len(df) > umbral:
        print(f"   AVISO: {n} filas ({n/len(df)*100:.1f}%) sin categoria en algun "
              f"one-hot; se dejan (son demasiadas para tratarlas como dato faltante)")
        return df
    print(f"   {n} filas sin categoria en algun one-hot descartadas del GAN "
          f"({n/len(df)*100:.2f}%, dato faltante)")
    return df[~incompletas].reset_index(drop=True)


def _informe_onehot(df: pd.DataFrame, grupos: Dict[str, List[str]], etiqueta: str) -> None:
    for prefijo, cols in grupos.items():
        suma = df[cols].sum(axis=1)
        rotas = int((suma > 1).sum())
        vacias = int((suma == 0).sum())
        print(f"   {etiqueta} {prefijo}: {len(df) - rotas - vacias}/{len(df)} con una sola "
              f"categoria, {rotas} con varias, {vacias} sin ninguna")

# --------- SANITIZACIÓN ---------
def _sanitize_for_sdv(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    # El GAN se entrena solo con CASOS COMPLETOS: se descartan filas con algun
    # valor faltante (p.ej. glucosa/HbA1c opcionales en NHANES) en vez de
    # imputarlas a 0, que le meteria al sintetico un pico artificial en cero. En
    # datasets ya imputados (Kaggle/ENSANUT) no hay NaN, asi que es un no-op.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if c == "target":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64")
        else:
            df[c] = df[c].astype(str)

    # Cap solo superior (valores fisiologicamente imposibles antes de ajustar)
    caps = {"blood_pressure": 300, "glucose": 500, "blood_glucose_level": 500, "bmi": 80}
    for col, hi in caps.items():
        if col in df.columns:
            x = df[col].values
            x = np.where(x > hi, hi, x)
            df[col] = x
    return df

# --------- SÍNTESIS ---------
def fit_and_sample_sdv(train_df, model, synth_multiplier, seed, epochs, max_train_rows, pasos_objetivo):
    if not SDV_AVAILABLE:
        raise RuntimeError("SDV no disponible. Revisa instalación.")

    np.random.seed(seed)
    df = _sanitize_for_sdv(train_df)

    # AUD-13: los grupos one-hot se detectan sobre los datos REALES y se colapsan a
    # una sola columna categorica ANTES de filtrar y entrenar. Asi el GAN modela
    # "genero" como una eleccion entre categorias excluyentes, no como dos monedas
    # independientes que pueden salir cara las dos.
    grupos_onehot = _detect_onehot_groups(df)
    if grupos_onehot:
        print(f"   one-hot colapsados para el GAN: "
              + ", ".join(f"{p} ({len(c)} cols)" for p, c in grupos_onehot.items()))
    df = _descartar_sin_categoria(df, grupos_onehot)
    df = _colapsar_onehot(df, grupos_onehot)

    cols_keep = []
    # Bloque NUEVO (Permite binarios como género y enfermedades)
    for c in df.columns:
        if c == "target" or c in grupos_onehot:
            cols_keep.append(c); continue

        # Guardamos todo lo que tenga al menos 2 valores distintos (binarios o continuos)
        if df[c].nunique() >= 2:
            cols_keep.append(c)
    df = df[sorted(set(cols_keep + ["target"]))]

    if max_train_rows and len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    # Las epocas se derivan del objetivo de pasos, ya con el numero final de filas.
    if epochs is None:
        epochs = _epocas_para(len(df), pasos_objetivo)
    pasos = epochs * max(1, math.ceil(len(df) / BATCH_SIZE_CTGAN))
    print(f"   entrenando el GAN: {epochs} epocas sobre {len(df)} filas (~{pasos} pasos)")

    if SDV_API_V1:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        for c in set(_detect_categorical_columns(df) + ["target"] + list(grupos_onehot)):
            if c in metadata.columns:
                metadata.update_column(c, sdtype="categorical")

        SynthClass = TVAESynthesizer if model.lower() == "tvae" else CTGANSynthesizer
        synthesizer = SynthClass(metadata, epochs=epochs, verbose=False)
        synthesizer.fit(df)
        n_synth = max(1, int(len(df) * synth_multiplier))
        muestra = synthesizer.sample(num_rows=n_synth).reset_index(drop=True)
    else:
        SynthClass = TVAE if model.lower() == "tvae" else CTGAN
        synthesizer = SynthClass(epochs=epochs, verbose=False)
        discretas = list(set(_detect_categorical_columns(df) + ["target"] + list(grupos_onehot)))
        synthesizer.fit(df, discrete_columns=discretas)
        n_synth = max(1, int(len(df) * synth_multiplier))
        muestra = synthesizer.sample(n_synth).reset_index(drop=True)

    # Vuelta al esquema real: la categorica se reparte en sus columnas one-hot.
    muestra = _expandir_onehot(muestra, grupos_onehot)
    # Mismo orden de columnas que el train, para que la ficha de paciente y las
    # comparaciones de distribucion no dependan del orden en que salio del GAN.
    orden = [c for c in train_df.columns if c in muestra.columns]
    return muestra[orden + [c for c in muestra.columns if c not in orden]]

# --------- PROCESO PRINCIPAL ---------
def process_one_dataset(name, test_size, seed, model, synth_multiplier, balance, epochs, max_train_rows,
                        pasos_objetivo):
    src = os.path.join(PROCESSED_DIR, FILENAME.format(name=name))
    if not os.path.exists(src):
        print(f"No existe {src}, se omite.")
        return
    df = pd.read_csv(src, low_memory=False)

    # AUD-12: sin esto, las filas repetidas caian a ambos lados del split (219 en
    # cardiovascular, 22 en hipertension) y el test quedaba optimista: el modelo
    # ya habia visto esa fila exacta en entrenamiento.
    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < antes:
        print(f"{name}: {antes - len(df)} filas duplicadas descartadas antes del split")

    out_dir = os.path.join(CURATED_DIR, name)
    ensure_dir(out_dir)

    train_df, test_df = stratified_split(df, test_size, seed)
    train_df.to_csv(os.path.join(out_dir, f"{name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{name}_test.csv"), index=False)

    try:
        synth_df = fit_and_sample_sdv(train_df, model, synth_multiplier, seed, epochs, max_train_rows,
                                      PASOS_POR_DATASET.get(name, pasos_objetivo))
        synth_df["target"] = pd.to_numeric(synth_df["target"], errors="coerce").fillna(0).astype(int)
        if balance:
            ones = synth_df[synth_df["target"] == 1]
            zeros = synth_df[synth_df["target"] == 0]
            n = min(len(ones), len(zeros))
            if n > 0:
                synth_df = pd.concat([ones.sample(n, random_state=seed), zeros.sample(n, random_state=seed)], ignore_index=True)
        # AUD-13: control de que el sintetico respeta los one-hot (antes salia ~46%
        # de filas validas en diabetes; ahora tiene que ser el 100%).
        grupos = _detect_onehot_groups(train_df)
        if grupos:
            _informe_onehot(synth_df, grupos, "sintetico")
            for prefijo, cols in grupos.items():
                rotas = int((synth_df[cols].sum(axis=1) > 1).sum())
                if rotas:
                    raise RuntimeError(
                        f"{rotas} filas con varias categorias de {prefijo} en el sintetico")

        synth_path = os.path.join(out_dir, f"{name}_synthetic_{model.lower()}_x{synth_multiplier:g}_seed{seed}.csv")
        synth_df.to_csv(synth_path, index=False)
        print(f"{name}: split + sintético ({model}, x{synth_multiplier}) guardado en {out_dir}")
    except Exception as e:
        print(f"{name}: no se generó sintético ({type(e).__name__}: {e})")

def main():
    parser = argparse.ArgumentParser(description="Curación, split y síntesis (GAN) por dataset")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="ctgan", choices=["ctgan", "tvae"])
    parser.add_argument("--synth_multiplier", type=float, default=1.0)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epocas fijas. Por defecto se derivan de --steps segun el "
                             "tamano del dataset (ver AUD-24).")
    parser.add_argument("--steps", type=int, default=15000,
                        help="Pasos de gradiente objetivo para el GAN (default 15000).")
    parser.add_argument("--max_train_rows", type=int, default=60000)
    parser.add_argument("--only", type=str, default="", help="Lista separada por comas de datasets a procesar")
    args = parser.parse_args()

    ensure_dir(CURATED_DIR)
    print(f"== Curación con test_size={args.test_size}, seed={args.seed}, model={args.model}, synth_multiplier={args.synth_multiplier}, balance={args.balance} ==")

    targets = DATASETS if not args.only else [s.strip() for s in args.only.split(",") if s.strip()]
    for name in targets:
        process_one_dataset(name, args.test_size, args.seed, args.model, args.synth_multiplier,
                            args.balance, args.epochs, args.max_train_rows, args.steps)

if __name__ == "__main__":
    main()