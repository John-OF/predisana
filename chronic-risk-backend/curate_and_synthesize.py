# curate_and_synthesize.py
import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List

# --------- RUTAS ---------
PROCESSED_DIR = "data_processed"
CURATED_DIR = "data_curated"

DATASETS = ["diabetes", "hipertension", "obesidad", "cardiovascular"]
FILENAME = "{name}_dataset.csv"  # dentro de data_processed

# --------- DICCIONARIO DE DESCRIPCIONES ---------
COLUMN_DESCRIPTIONS = {
    "age": "Edad del paciente en años.",
    "gender": "Género biológico del paciente.",
    "gender_Female": "Variable dummy: Género Femenino (0=No, 1=Sí).",
    "gender_Male": "Variable dummy: Género Masculino (0=No, 1=Sí).",
    "height": "Altura del paciente.",
    "weight": "Peso del paciente.",
    "bmi": "Índice de Masa Corporal (kg/m²).",
    "glucose": "Nivel de glucosa en sangre (mg/dL).",
    "blood_glucose_level": "Nivel de glucosa en sangre (copia para compatibilidad).",
    "hba1c_level": "Nivel de Hemoglobina Glicosilada (%).",
    "blood_pressure": "Presión arterial (sistólica/diastólica unificada o predominante).",
    "insulin": "Nivel de insulina sérica (mu U/ml).",
    "pregnancies": "Número de embarazos previos.",
    "skin_thickness": "Grosor del pliegue cutáneo del tríceps (mm).",
    "diabetes_pedigree": "Función de pedigrí de diabetes (historial genético).",
    "hypertension": "Historial de hipertensión (0=No, 1=Sí).",
    "heart_disease": "Historial de enfermedad cardíaca (0=No, 1=Sí).",
    "smoking_history": "Historial de tabaquismo (categórico).",
    "smoking_history_current": "Fumador actual.",
    "smoking_history_never": "Nunca ha fumado.",
    "smoking_history_former": "Ex-fumador.",
    "smoking_history_ever": "Alguna vez ha fumado.",
    "smoking_history_not current": "No fuma actualmente.",
    "target": "Variable objetivo (Clase a predecir: 0=Negativo, 1=Positivo)."
}

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

def _get_sql_type(dtype):
    """Mapea tipos de Pandas a tipos SQL genéricos para la documentación."""
    s = str(dtype).lower()
    if "int" in s: return "INT"
    if "float" in s: return "DECIMAL"
    if "object" in s or "string" in s: return "VARCHAR"
    if "bool" in s: return "BOOLEAN"
    return "UNKNOWN"

# --------- GENERACIÓN DE DICCIONARIO ---------
def create_data_dictionary(df: pd.DataFrame, dest_folder: str, dataset_name: str):
    """
    Genera el diccionario de datos con el formato solicitado por el tutor:
    Nombre del Campo | Tipo de Dato | Longitud | Descripción | Restricción | Ejemplo
    """
    rows = []
    
    for col in df.columns:
        s = df[col]
        dtype_sql = _get_sql_type(s.dtype)
        
        # 1. Longitud
        # Para texto calculamos el max, para números ponemos N/A
        length = "N/A"
        if dtype_sql == "VARCHAR":
            # Calcular longitud máxima real encontrada
            max_len = s.astype(str).str.len().max()
            length = str(max_len)
            
        # 2. Descripción
        # Buscamos en el mapa, si no existe, ponemos algo genérico
        desc = COLUMN_DESCRIPTIONS.get(col, f"Variable asociada a {col}.")
        
        # 3. Restricción
        constraints = []
        # Not Null
        if s.isna().sum() == 0:
            constraints.append("Not Null")
        
        # Tipos de valores
        if _is_binary(s):
            constraints.append("Binario (0, 1)")
        elif dtype_sql in ["INT", "DECIMAL"]:
            # Si es numérico mostramos rango
            vmin = float(s.min()) if not s.empty else 0
            vmax = float(s.max()) if not s.empty else 0
            constraints.append(f"Rango: [{vmin:.1f} - {vmax:.1f}]")
        
        if col == "id": # Si hubiera ID
            constraints.append("PK")
            
        restriction_str = ", ".join(constraints)

        # 4. Ejemplo
        example_val = s.dropna().iloc[0] if not s.dropna().empty else "N/A"
        
        entry = {
            "Nombre del Campo": col,
            "Tipo de Dato": dtype_sql,
            "Longitud": length,
            "Descripción": desc,
            "Restricción": restriction_str,
            "Ejemplo": str(example_val)
        }
        rows.append(entry)

    dd = pd.DataFrame(rows)
    
    # Guardar CSV
    dd_csv = os.path.join(dest_folder, f"data_dictionary_{dataset_name}.csv")
    dd.to_csv(dd_csv, index=False)

    # Guardar Markdown
    dd_md = os.path.join(dest_folder, f"data_dictionary_{dataset_name}.md")
    with open(dd_md, "w", encoding="utf-8") as f:
        f.write(f"# Diccionario de Datos: {dataset_name.capitalize()}\n\n")
        f.write(f"**Total de registros:** {len(df)}\n\n")
        # Usamos to_markdown de pandas si está disponible, sino manual
        try:
            f.write(dd.to_markdown(index=False))
        except ImportError:
            # Fallback manual si falta tabulate
            f.write("| " + " | ".join(dd.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(dd.columns)) + " |\n")
            for _, row in dd.iterrows():
                f.write("| " + " | ".join([str(x) for x in row.values]) + " |\n")

    print(f" Diccionario generado en: {dd_csv}")

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
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if c == "target":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64")
        else:
            df[c] = df[c].astype(str)

    # Cap solo superior
    caps = {"blood_pressure": 300, "glucose": 500, "bmi": 80}
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

    out_dir = os.path.join(CURATED_DIR, name)
    ensure_dir(out_dir)
    
    # AQUI SE LLAMA A LA NUEVA FUNCIÓN
    create_data_dictionary(df, out_dir, name)

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