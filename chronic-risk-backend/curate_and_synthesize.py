# curate_and_synthesize.py
import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List

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
def fit_and_sample_sdv(train_df, model, synth_multiplier, seed, epochs, max_train_rows):
    if not SDV_AVAILABLE:
        raise RuntimeError("SDV no disponible. Revisa instalación.")

    np.random.seed(seed)
    df = _sanitize_for_sdv(train_df)

    cols_keep = []
    # Bloque NUEVO (Permite binarios como género y enfermedades)
    for c in df.columns:
        if c == "target":
            cols_keep.append(c); continue
        
        # Guardamos todo lo que tenga al menos 2 valores distintos (binarios o continuos)
        if df[c].nunique() >= 2:
            cols_keep.append(c)
    df = df[sorted(set(cols_keep + ["target"]))]

    if max_train_rows and len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    if SDV_API_V1:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        for c in set(_detect_categorical_columns(df) + ["target"]):
            if c in metadata.columns:
                metadata.update_column(c, sdtype="categorical")

        SynthClass = TVAESynthesizer if model.lower() == "tvae" else CTGANSynthesizer
        synthesizer = SynthClass(metadata, epochs=epochs, verbose=False)
        synthesizer.fit(df)
        n_synth = max(1, int(len(df) * synth_multiplier))
        return synthesizer.sample(num_rows=n_synth).reset_index(drop=True)
    else:
        SynthClass = TVAE if model.lower() == "tvae" else CTGAN
        synthesizer = SynthClass(epochs=epochs, verbose=False)
        synthesizer.fit(df, discrete_columns=list(set(_detect_categorical_columns(df) + ["target"])))
        n_synth = max(1, int(len(df) * synth_multiplier))
        return synthesizer.sample(n_synth).reset_index(drop=True)

# --------- PROCESO PRINCIPAL ---------
def process_one_dataset(name, test_size, seed, model, synth_multiplier, balance, epochs, max_train_rows):
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
        synth_df = fit_and_sample_sdv(train_df, model, synth_multiplier, seed, epochs, max_train_rows)
        synth_df["target"] = pd.to_numeric(synth_df["target"], errors="coerce").fillna(0).astype(int)
        if balance:
            ones = synth_df[synth_df["target"] == 1]
            zeros = synth_df[synth_df["target"] == 0]
            n = min(len(ones), len(zeros))
            if n > 0:
                synth_df = pd.concat([ones.sample(n, random_state=seed), zeros.sample(n, random_state=seed)], ignore_index=True)
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=60000)
    parser.add_argument("--only", type=str, default="", help="Lista separada por comas de datasets a procesar")
    args = parser.parse_args()

    ensure_dir(CURATED_DIR)
    print(f"== Curación con test_size={args.test_size}, seed={args.seed}, model={args.model}, synth_multiplier={args.synth_multiplier}, balance={args.balance} ==")

    targets = DATASETS if not args.only else [s.strip() for s in args.only.split(",") if s.strip()]
    for name in targets:
        process_one_dataset(name, args.test_size, args.seed, args.model, args.synth_multiplier, args.balance, args.epochs, args.max_train_rows)

if __name__ == "__main__":
    main()