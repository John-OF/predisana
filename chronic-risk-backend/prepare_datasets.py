# prepare_datasets.py
# =============================================================================
# Limpieza de datos POR ENFERMEDAD (arquitectura v2, B1).
#
# Cambio respecto a la versión anterior:
#   - ANTES: se concatenaban ~7 CSVs heterogéneos en UN frame maestro y se
#     re-derivaba el target de cada enfermedad sobre las MISMAS filas. Eso
#     producía una "sopa imputada" (insulina/pliegue/pedigree 99.8% en cero,
#     presión 87.8% imputada, etc.) y modelos sin señal real.
#   - AHORA: cada enfermedad se limpia desde SU PROPIA fuente, con SU propio
#     esquema de columnas. Sin COMMON_SCHEMA global, sin imputación cruzada.
#     La imputación (cuando ocurre) es por mediana DENTRO de un único dataset
#     real y coherente, no entre fuentes distintas.
#
# Fuentes elegidas (ver BITACORA.md → B1):
#   - cardiovascular  -> data_raw/cardio_train.csv                  (70k, sep=';')
#   - hipertension    -> data_raw/Hipertension_Arterial_Mexico.csv  (4.4k, ENSANUT)
#
# DIABETES NO SE PROCESA AQUÍ: migró a NHANES 2021-2023 (datos reales de los CDC,
# glucosa continua). Su dataset lo genera prepare_nhanes_diabetes.py y su modelo
# híbrido lo entrena train_nhanes_diabetes.py. El flujo Kaggle viejo (glucosa
# cuantizada, target escalonado) se retiró para que correr este script no pise
# el dataset NHANES vivo.
#
# Descartados a propósito:
#   - data_raw/hypertension_dataset.csv  -> RUIDO (target aleatorio, corr ~0.00)
#   - data_raw/diabetes.csv (PIMA) y Dataset_of_Diabetes.csv -> metían imputación
#   - data_raw/ObesityDataSet_*.csv -> obesidad fuera del alcance (A1)
# =============================================================================

import os
import pandas as pd
import numpy as np

RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =============================================================================
# Utilidades comunes
# =============================================================================
def _read_csv(path, sep=","):
    """Lectura robusta (utf-8 -> latin-1)."""
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1")


def _clip(series, lo, hi):
    """Recorta a un rango fisiológico plausible; fuera de rango -> NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= lo) & (s <= hi), np.nan)


def _impute_median(df, cols):
    """Imputa por mediana DENTRO de este único dataset real (no cruzada)."""
    for c in cols:
        if c in df.columns:
            med = df[c].median()
            n_na = int(df[c].isna().sum())
            if n_na > 0 and pd.notna(med):
                df[c] = df[c].fillna(med)
                print(f"      -{c}: {n_na} nulos -> mediana {med:.2f}")
    return df


def _report(name, df, target="target"):
    n = len(df)
    pos = int(df[target].sum())
    na = int(df.isna().sum().sum())
    print(f"   => {name}: filas={n}  positivos={pos} ({100*pos/n:.1f}%)  "
          f"cols={df.shape[1]}  NaN_restantes={na}")


# =============================================================================
# CARDIOVASCULAR  <- cardio_train.csv (70k, sep=';')
# Columns: id, age(días), gender(1/2), height(cm), weight(kg), ap_hi, ap_lo,
#          cholesterol(1-3), gluc(1-3), smoke, alco, active, cardio
# =============================================================================
def build_cardiovascular():
    print("\n=== CARDIOVASCULAR (cardio_train.csv) ===")
    src = _read_csv(os.path.join(RAW_DIR, "cardio_train.csv"), sep=";")
    src.columns = [c.strip().lower() for c in src.columns]

    out = pd.DataFrame()
    # Edad viene en DÍAS -> años
    out["age"] = (pd.to_numeric(src["age"], errors="coerce") / 365.25).round(1)

    # IMC derivado de altura/peso (con saneamiento)
    h_cm = _clip(src["height"], 120, 220)        # cm plausibles
    w_kg = _clip(src["weight"], 30, 250)         # kg plausibles
    bmi = w_kg / (h_cm / 100.0) ** 2
    out["bmi"] = _clip(bmi, 12, 70)

    # Presión arterial: el crudo trae errores groseros (ap_hi hasta 16020, negativos)
    out["ap_hi"] = _clip(src["ap_hi"], 70, 250)  # sistólica
    out["ap_lo"] = _clip(src["ap_lo"], 40, 150)  # diastólica

    # Ordinales clínicos (1=normal, 2=alto, 3=muy alto) — se preservan como orden
    out["cholesterol"] = pd.to_numeric(src["cholesterol"], errors="coerce")
    out["gluc"] = pd.to_numeric(src["gluc"], errors="coerce")

    # Binarias de estilo de vida
    out["smoke"] = pd.to_numeric(src["smoke"], errors="coerce").fillna(0).astype(int)
    out["alco"] = pd.to_numeric(src["alco"], errors="coerce").fillna(0).astype(int)
    out["active"] = pd.to_numeric(src["active"], errors="coerce").fillna(0).astype(int)

    # Género (cardio dataset: 1=mujer, 2=hombre)
    out["gender_Female"] = (pd.to_numeric(src["gender"], errors="coerce") == 1).astype(int)
    out["gender_Male"] = (pd.to_numeric(src["gender"], errors="coerce") == 2).astype(int)

    out["target"] = pd.to_numeric(src["cardio"], errors="coerce").fillna(0).astype(int)

    # Descartar filas con presión incoherente (diastólica >= sistólica) o presión perdida
    before = len(out)
    out = out[out["ap_hi"].notna() & out["ap_lo"].notna()]
    out = out[out["ap_hi"] > out["ap_lo"]]
    print(f"      -presión incoherente/ausente descartada: {before - len(out)} filas")

    out = _impute_median(out, ["age", "bmi", "cholesterol", "gluc"])
    out = out.dropna(subset=["target"]).reset_index(drop=True)

    _report("cardiovascular", out)
    return out


# =============================================================================
# HIPERTENSIÓN  <- Hipertension_Arterial_Mexico.csv (ENSANUT, 4.4k)
# Survey con 36 columnas; seleccionamos las con señal y descartamos el resto.
# Target: riesgo_hipertension (0/1).
# =============================================================================
def build_hipertension():
    print("\n=== HIPERTENSIÓN (Hipertension_Arterial_Mexico.csv) ===")
    src = _read_csv(os.path.join(RAW_DIR, "Hipertension_Arterial_Mexico.csv"))
    src.columns = [c.strip().lower() for c in src.columns]

    # Mapa columna_origen -> nombre_destino (solo variables con sentido clínico)
    colmap = {
        "edad": "age",
        "masa_corporal": "bmi",
        "peso": "weight",
        "medida_cintura": "waist_circumference",
        "tension_arterial": "blood_pressure",
        "resultado_glucosa": "glucose",
        "valor_hemoglobina_glucosilada": "hba1c_level",
        "valor_colesterol_total": "cholesterol_total",
        "valor_colesterol_hdl": "hdl",
        "valor_colesterol_ldl": "ldl",
        "valor_trigliceridos": "triglycerides",
        "valor_insulina": "insulin",
    }

    out = pd.DataFrame()
    for srccol, dst in colmap.items():
        if srccol in src.columns:
            out[dst] = pd.to_numeric(src[srccol], errors="coerce")

    # Saneamiento de rangos fisiológicos
    if "age" in out: out["age"] = _clip(out["age"], 0, 120)
    if "bmi" in out: out["bmi"] = _clip(out["bmi"], 12, 70)
    if "blood_pressure" in out: out["blood_pressure"] = _clip(out["blood_pressure"], 70, 260)
    if "glucose" in out: out["glucose"] = _clip(out["glucose"], 40, 500)
    if "hba1c_level" in out: out["hba1c_level"] = _clip(out["hba1c_level"], 3.0, 18.0)
    if "waist_circumference" in out: out["waist_circumference"] = _clip(out["waist_circumference"], 40, 200)
    if "weight" in out: out["weight"] = _clip(out["weight"], 25, 250)

    # Descartar columnas demasiado vacías (>40% NaN) para no reintroducir "sopa"
    keep = []
    for c in out.columns:
        na_rate = out[c].isna().mean()
        if na_rate > 0.40:
            print(f"      -descartada '{c}' por {100*na_rate:.0f}% NaN")
        else:
            keep.append(c)
    out = out[keep]

    # Género (sexo: 1=hombre, 2=mujer según el survey)
    if "sexo" in src.columns:
        sexo = pd.to_numeric(src["sexo"], errors="coerce")
        out["gender_Male"] = (sexo == 1).astype(int)
        out["gender_Female"] = (sexo == 2).astype(int)

    # Target
    rh = pd.to_numeric(src["riesgo_hipertension"], errors="coerce")
    out["target"] = (rh >= 0.5).astype(int)

    # Imputación por mediana dentro de este único dataset real
    num_cols = [c for c in out.columns if c not in ("target", "gender_Male", "gender_Female")]
    out = _impute_median(out, num_cols)
    out = out.dropna(subset=["target"]).reset_index(drop=True)

    _report("hipertension", out)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Diabetes NO está aquí a propósito: su dataset canónico es NHANES y lo
    # genera prepare_nhanes_diabetes.py (correr este script no debe pisarlo).
    builders = {
        "cardiovascular": build_cardiovascular,
        "hipertension": build_hipertension,
    }
    print("Generando datasets limpios POR ENFERMEDAD (sin frame maestro)...")
    for name, fn in builders.items():
        df = fn()
        path = os.path.join(PROCESSED_DIR, f"{name}_dataset.csv")
        df.to_csv(path, index=False)
        print(f"   guardado -> {path}")

    print("\nListo. Cada enfermedad tiene su propio esquema, sin imputación cruzada.")
    print("(Diabetes va aparte: python prepare_nhanes_diabetes.py)")


if __name__ == "__main__":
    main()
