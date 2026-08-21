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
#
# DIABETES E HIPERTENSIÓN NO SE PROCESAN AQUÍ: las dos migraron a NHANES 2021-2023
# (datos reales de los CDC). Sus datasets los generan prepare_nhanes_diabetes.py y
# prepare_nhanes_hipertension.py. Los flujos viejos se retiraron de este script para
# que correrlo NO pise los datasets NHANES vivos:
#   - Kaggle de diabetes: glucosa cuantizada y target escalonado.
#   - ENSANUT de hipertensión (AUD-1): el target `riesgo_hipertension` era una
#     fórmula del autor del CSV, no un desenlace; el modelo la reaprendía y salían
#     relaciones invertidas (a más edad, menos riesgo).
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
# MAIN
# =============================================================================
def main():
    # Diabetes e hipertensión NO están aquí a propósito: sus datasets canónicos
    # son NHANES y los generan prepare_nhanes_*.py (correr esto no debe pisarlos).
    builders = {
        "cardiovascular": build_cardiovascular,
    }
    print("Generando datasets limpios POR ENFERMEDAD (sin frame maestro)...")
    for name, fn in builders.items():
        df = fn()
        path = os.path.join(PROCESSED_DIR, f"{name}_dataset.csv")
        df.to_csv(path, index=False)
        print(f"   guardado -> {path}")

    print("\nListo. Cada enfermedad tiene su propio esquema, sin imputación cruzada.")
    print("(Diabetes e hipertensión van aparte: python prepare_nhanes_diabetes.py"
          " / python prepare_nhanes_hipertension.py)")


if __name__ == "__main__":
    main()
