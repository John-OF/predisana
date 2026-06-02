# prepare_datasets.py
import os
import pandas as pd
import numpy as np

# ====== CONFIG ======
RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Archivos CSV esperados
CSV_FILES = [
    "diabetes.csv",
    "diabetes_prediction_dataset.csv",
    "Dataset_of_Diabetes.csv",
    "Hipertension_Arterial_Mexico.csv",
    "hypertension_dataset.csv",
    "ObesityDataSet_raw_and_data_sinthetic.csv",
    "cardio_train.csv",
]

# Esquema común.
# NOTA: Mantenemos 'blood_glucose_level' para compatibilidad con train_models,
# pero será una copia exacta de 'glucose'.
COMMON_SCHEMA = [
    "age",
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "hypertension",
    "heart_disease",
    "hba1c_level",
    "blood_glucose_level", 
    "gender_Female",
    "gender_Male",
    "smoking_history_current",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_ever",
    "smoking_history_not current",
    "target"
]

# Valores por defecto médicos (SÓLO para casos de emergencia donde falte TODA la columna)
MEDICAL_DEFAULTS = {
    "glucose": 100.0,
    "hba1c_level": 5.5,
    "bmi": 25.0,
    "blood_pressure": 120.0,
    "age": 40.0
}

# =========================================================
# 1. CARGA
# =========================================================
def load_csvs():
    loaded = []
    for name in CSV_FILES:
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            print(f"No existe: {name}")
            continue

        df = None
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding="latin-1")
            except:
                pass
        
        # Intento con separador ;
        if df is not None and df.shape[1] <= 1:
            try:
                df = pd.read_csv(path, sep=";", encoding="latin-1")
            except:
                pass

        if df is None:
            print(f"No se pudo cargar: {name}")
            continue

        print(f"Cargado: {name} -> {df.shape}")
        loaded.append((name, df))
    return loaded

# =========================================================
# 2. DETECTORES
# =========================================================
def is_pima_like(df):
    needed = {"Pregnancies", "Glucose", "BloodPressure", "BMI", "Age", "Outcome"}
    return needed.issubset(df.columns)

def is_prediction_like(df):
    lower = {c.lower() for c in df.columns}
    return {"age", "bmi", "hypertension"}.issubset(lower) and ("diabetes" in lower)

def is_lab_diabetes_like(df):
    cols = {c.lower().strip() for c in df.columns}
    return ("class" in cols or "clas" in cols or "status" in cols) and \
           any(x in cols for x in ["hba1c", "hb1ac", "valor_hemoglobina_glucosilada"])

def is_mexican_hypertension(df):
    cols = {c.lower().strip() for c in df.columns}
    return "riesgo_hipertension" in cols or ("tension_arterial" in cols and "edad" in cols)

def is_english_hypertension(df):
    cols = {c.lower().strip() for c in df.columns}
    return "hypertension" in cols and ("age" in cols or "bmi" in cols)

def is_obesity_uci(df):
    cols = {c.lower().strip() for c in df.columns}
    return "nobeyesdad" in cols or "obesity" in cols

def is_cardio_kaggle(df):
    cols = {c.lower().strip() for c in df.columns}
    return "cardio" in cols and "ap_hi" in cols

# =========================================================
# 3. NORMALIZADORES
# =========================================================

def init_df(rows):
    """Crea un DF vacío con NaNs en lugar de ceros"""
    df = pd.DataFrame(np.nan, index=range(rows), columns=COMMON_SCHEMA)
    # Las dummies sí pueden ser 0
    dummies = [c for c in COMMON_SCHEMA if "gender" in c or "smoking" in c or c in ["hypertension", "heart_disease", "target"]]
    df[dummies] = 0
    return df

def normalize_pima(df):
    out = init_df(len(df))
    
    out["pregnancies"] = df["Pregnancies"]
    out["glucose"] = df["Glucose"].replace(0, np.nan) # Corregir ceros fisiológicos
    out["blood_pressure"] = df["BloodPressure"].replace(0, np.nan)
    out["skin_thickness"] = df["SkinThickness"].replace(0, np.nan)
    out["insulin"] = df["Insulin"].replace(0, np.nan)
    out["bmi"] = df["BMI"].replace(0, np.nan)
    out["diabetes_pedigree"] = df["DiabetesPedigreeFunction"]
    out["age"] = df["Age"]
    out["target"] = df["Outcome"]
    
    # PIMA no tiene HbA1c, se deja como NaN para imputar luego
    return out

def normalize_prediction(df):
    # Convertir columnas a minúsculas y quitar espacios
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))

    # Columnas básicas
    out["age"] = df["age"] if "age" in df else np.nan
    out["bmi"] = df["bmi"] if "bmi" in df else np.nan
    
    # Asignaciones SEGURAS (Evitan el KeyError 'heart_disease')
    out["hypertension"] = df["hypertension"] if "hypertension" in df else 0
    out["heart_disease"] = df["heart_disease"] if "heart_disease" in df else 0
    out["hba1c_level"] = df["hba1c_level"] if "hba1c_level" in df else np.nan
    
    # Glucosa: puede venir como blood_glucose_level o no venir
    if "blood_glucose_level" in df:
        out["glucose"] = df["blood_glucose_level"]
    elif "glucose" in df:
        out["glucose"] = df["glucose"]
    else:
        out["glucose"] = np.nan

    # Target
    if "diabetes" in df:
        out["target"] = df["diabetes"]
    else:
        out["target"] = 0

    # Gender
    if "gender" in df:
        g = df["gender"].astype(str).str.lower()
        out["gender_Female"] = (g == "female").astype(int)
        out["gender_Male"] = (g == "male").astype(int)

    # Smoking
    if "smoking_history" in df:
        s = df["smoking_history"].astype(str).str.lower()
        out["smoking_history_current"] = (s == "current").astype(int)
        out["smoking_history_former"] = (s == "former").astype(int)
        out["smoking_history_never"] = (s == "never").astype(int)
        out["smoking_history_ever"] = ((s == "current") | (s == "former") | (s == "ever")).astype(int) 
        out["smoking_history_not current"] = (s == "not current").astype(int)

    return out

def normalize_lab_diabetes(df):
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))
    
    out["age"] = df["age"] if "age" in df else np.nan
    out["bmi"] = df["bmi"] if "bmi" in df else np.nan
    
    # HbA1c
    if "hba1c" in df: out["hba1c_level"] = df["hba1c"]
    elif "valor_hemoglobina_glucosilada" in df: out["hba1c_level"] = df["valor_hemoglobina_glucosilada"]
    
    # Glucose (Este dataset a veces no tiene glucosa, dejar NaN)
    # Gender
    if "gender" in df:
        g = df["gender"].astype(str).str.lower()
        out["gender_Female"] = g.str.startswith("f").astype(int)
        out["gender_Male"] = g.str.startswith("m").astype(int)

    # Target
    if "class" in df:
        out["target"] = (df["class"].astype(str).str.upper() != "N").astype(int)
    elif "status" in df:
        out["target"] = (df["status"].astype(str).str.lower() != "normal").astype(int)

    return out

def normalize_mexican_hypertension(df):
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))

    out["age"] = df["edad"] if "edad" in df else np.nan
    out["bmi"] = df["masa_corporal"] if "masa_corporal" in df else np.nan
    out["glucose"] = df["resultado_glucosa"] if "resultado_glucosa" in df else np.nan
    out["hba1c_level"] = df["valor_hemoglobina_glucosilada"] if "valor_hemoglobina_glucosilada" in df else np.nan

    if "sexo" in df:
        # Asumiendo 1=H, 2=M según tu código anterior
        out["gender_Male"] = (df["sexo"] == 1).astype(int)
        out["gender_Female"] = (df["sexo"] == 2).astype(int)

    if "riesgo_hipertension" in df:
        rh = pd.to_numeric(df["riesgo_hipertension"], errors='coerce').fillna(0)
        out["hypertension"] = (rh == 1).astype(int)
        out["target"] = out["hypertension"]

    return out

def normalize_english_hypertension(df):
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))
    
    out["age"] = df["age"] if "age" in df else np.nan
    out["bmi"] = df["bmi"] if "bmi" in df else np.nan
    out["glucose"] = df["glucose"] if "glucose" in df else np.nan
    
    if "gender" in df:
        g = df["gender"].astype(str).str.lower()
        out["gender_Female"] = (g == "female").astype(int)
        out["gender_Male"] = (g == "male").astype(int)

    if "hypertension" in df:
        h = df["hypertension"].astype(str).str.lower()
        out["hypertension"] = h.apply(lambda x: 1 if x in ["1","yes","true","high"] else 0)
        out["target"] = out["hypertension"]
        
    return out

def normalize_obesity_uci(df):
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))

    out["age"] = df["age"] if "age" in df else np.nan
    
    # Calcular BMI
    if "height" in df and "weight" in df:
        h = df["height"].astype(float)
        w = df["weight"].astype(float)
        if h.mean() > 3: h = h / 100.0 # Ajuste cm a metros
        out["bmi"] = w / (h**2)

    if "gender" in df:
        g = df["gender"].astype(str).str.lower()
        out["gender_Female"] = g.str.startswith("f").astype(int)
        out["gender_Male"] = g.str.startswith("m").astype(int)

    # Target
    if "nobeyesdad" in df:
        cat = df["nobeyesdad"].astype(str).str.lower()
        out["target"] = cat.apply(lambda x: 1 if "overweight" in x or "obesity" in x else 0)
    else:
        out["target"] = (out["bmi"] >= 30).astype(int)

    return out

def normalize_cardio_kaggle(df):
    df.columns = [c.strip().lower() for c in df.columns]
    out = init_df(len(df))

    # Edad viene en días
    if "age" in df:
        out["age"] = (df["age"] / 365.25).round(1)

    # BMI
    if "height" in df and "weight" in df:
        h = df["height"].astype(float) / 100.0
        w = df["weight"].astype(float)
        out["bmi"] = w / (h**2)

    if "ap_hi" in df:
        out["blood_pressure"] = df["ap_hi"].replace(0, np.nan)

    # Glucosa (1,2,3 -> mapear a valores reales)
    if "gluc" in df:
        out["glucose"] = df["gluc"].map({1: 90, 2: 130, 3: 180})

    if "gender" in df:
        # Cardio dataset: 1=Female, 2=Male usualmente, a veces al revés, 
        # asumiremos 1=Female por consistencia con tu script previo
        out["gender_Female"] = (df["gender"] == 1).astype(int)
        out["gender_Male"] = (df["gender"] == 2).astype(int)
    
    if "smoke" in df:
        out["smoking_history_current"] = df["smoke"].fillna(0).astype(int)
        out["smoking_history_ever"] = df["smoke"].fillna(0).astype(int)

    if "cardio" in df:
        out["target"] = df["cardio"]

    return out

def normalize_unknown(df):
    return init_df(len(df))

# =========================================================
# 4. POST-PROCESAMIENTO E IMPUTACIÓN
# =========================================================
def fill_missing_values(df):
    """
    Rellena NaNs con la Mediana de la columna.
    Si la columna está vacía entera, usa MEDICAL_DEFAULTS.
    """
    print("Ejecutando imputación de datos faltantes...")
    
    physiological_cols = ["glucose", "hba1c_level", "bmi", "blood_pressure", "age"]
    
    for col in physiological_cols:
        # 1. Si hay ceros donde no debería, convertirlos a NaN
        if col in df.columns:
            # Solo tratar 0 como NaN en variables continuas, no binarias
            count_zeros = (df[col] == 0).sum()
            if count_zeros > 0:
                print(f"   - {col}: {count_zeros} ceros convertidos a NaN")
                df[col] = df[col].replace(0, np.nan)

            # 2. Calcular mediana disponible
            median_val = df[col].median()
            
            # 3. Si es NaN (toda la columna vacía), usar default global
            if pd.isna(median_val):
                median_val = MEDICAL_DEFAULTS.get(col, 0)
                print(f"   - {col}: Vacía. Usando default médico -> {median_val}")
            else:
                print(f"   - {col}: Rellenando nulos con Mediana -> {median_val:.2f}")

            # 4. Rellenar
            df[col] = df[col].fillna(median_val)
    
    return df

def finalize_schema(df):
    # Asegurar tipos
    df = df.copy()
    
    # Clonar glucose a blood_glucose_level para compatibilidad con train_models.py
    df["blood_glucose_level"] = df["glucose"]
    
    # Asegurar que no quedan NaNs
    df = df.fillna(0)
    
    return df

# =========================================================
# 5. MAIN
# =========================================================
def main():
    loaded = load_csvs()
    normalized = []

    for name, df in loaded:
        if is_pima_like(df):
            print(f"➡️ {name} es PIMA")
            nd = normalize_pima(df)
        elif is_prediction_like(df):
            print(f"➡️ {name} es PREDICTION")
            nd = normalize_prediction(df)
        elif is_lab_diabetes_like(df):
            print(f"➡️ {name} es LAB-DIABETES")
            nd = normalize_lab_diabetes(df)
        elif is_mexican_hypertension(df):
            print(f"➡️ {name} es HIPERTENSION-MX")
            nd = normalize_mexican_hypertension(df)
        elif is_english_hypertension(df):
            print(f"➡️ {name} es HIPERTENSION-EN")
            nd = normalize_english_hypertension(df)
        elif is_obesity_uci(df):
            print(f"➡️ {name} es OBESIDAD-UCI")
            nd = normalize_obesity_uci(df)
        elif is_cardio_kaggle(df):
            print(f"➡️ {name} es CARDIO-KAGGLE")
            nd = normalize_cardio_kaggle(df)
        else:
            print(f"➡️ {name} es DESCONOCIDO")
            nd = normalize_unknown(df)

        normalized.append(nd)

    if not normalized:
        print("No se generaron datos.")
        return

    # Unir todo en un gran dataset maestro
    full_df = pd.concat(normalized, ignore_index=True)
    
    # Imputar valores faltantes
    full_df = fill_missing_values(full_df)
    full_df = finalize_schema(full_df)

    # Forzar que el 'target' sea numérico antes de usarlo
    # Si hay texto ("positive", "yes", etc) que se nos pasó, esto lo intenta convertir.
    # Si falla, pone NaN y luego 0.
    full_df["target"] = pd.to_numeric(full_df["target"], errors='coerce').fillna(0)

    print(f"\n Dataset Maestro Generado: {full_df.shape}")
    print(full_df.describe().loc[['min', 'max', 'mean']])

    # =====================================================
    # GENERACIÓN DE DATASETS ESPECÍFICOS
    # =====================================================

    # 1. Diabetes (Target original)
    df_diab = full_df.copy()
    # Ahora ya es seguro comparar con 0.5 porque garantizamos que es numérico
    df_diab["target"] = (df_diab["target"] >= 0.5).astype(int)
    df_diab.to_csv(os.path.join(PROCESSED_DIR, "diabetes_dataset.csv"), index=False)
    print("diabetes_dataset.csv guardado.")

    # 2. Hipertensión (Target = columna hypertension)
    df_hyp = full_df.copy()
    # Aseguramos también que hypertension sea entero
    df_hyp["hypertension"] = pd.to_numeric(df_hyp["hypertension"], errors='coerce').fillna(0)
    df_hyp["target"] = (df_hyp["hypertension"] >= 0.5).astype(int)
    df_hyp.to_csv(os.path.join(PROCESSED_DIR, "hipertension_dataset.csv"), index=False)
    print("hipertension_dataset.csv guardado.")

    # 3. Obesidad (Target = BMI >= 30)
    df_obe = full_df.copy()
    df_obe["target"] = (df_obe["bmi"] >= 30).astype(int)
    df_obe.to_csv(os.path.join(PROCESSED_DIR, "obesidad_dataset.csv"), index=False)
    print("obesidad_dataset.csv guardado.")

    # 4. Cardiovascular (Target = heart_disease O original)
    df_cardio = full_df.copy()
    
    # Asegurar numéricos también aquí
    df_cardio["heart_disease"] = pd.to_numeric(df_cardio["heart_disease"], errors='coerce').fillna(0)
    
    # Lógica: Si heart_disease es 1 O el target original era 1 (y cardio es relevante), es enfermo.
    df_cardio["target"] = ((df_cardio["heart_disease"] == 1) | (df_cardio["target"] == 1)).astype(int)
    
    df_cardio.to_csv(os.path.join(PROCESSED_DIR, "cardiovascular_dataset.csv"), index=False)
    print("cardiovascular_dataset.csv guardado.")

if __name__ == "__main__":
    main()