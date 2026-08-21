# synthetic_quality.py
# Calidad del sintetico vs real: score SDMetrics + matrices de correlacion.
#
# Vive fuera de app.py porque tiene DOS consumidores:
#   - build_quality_reports.py (pipeline, offline) -> precomputa el JSON.
#   - app.py (runtime)         -> sirve ese JSON y solo recalcula como fallback.
#
# POR QUE precomputar (AUD-17): `sdmetrics` arrastra torch (~479 MB) al importar
# QualityReport, y el API no puede pagar eso en el deploy. Con el JSON generado
# en el pipeline, produccion no necesita ni sdv ni sdmetrics, y el endpoint
# responde al instante en vez de calcular varios segundos en el primer hit.
import glob
import json
import os

import pandas as pd

CURATED_DIR = "data_curated"

# Variables continuas por enfermedad para el heatmap de correlaciones.
CORR_FEATURES = {
    "diabetes": ["age", "bmi", "blood_glucose_level", "hba1c_level"],
    "hipertension": ["age", "bmi", "weight", "waist_circumference"],
    "cardiovascular": ["age", "bmi", "ap_hi", "ap_lo"],
}


def report_path(disease: str) -> str:
    return os.path.join(CURATED_DIR, disease, f"{disease}_quality.json")


def load_precomputed(disease: str):
    """Informe generado por el pipeline. None si no existe o esta corrupto."""
    path = report_path(disease)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_precomputed"] = True
        return data
    except (ValueError, OSError) as e:
        print(f"[WARN] informe de calidad ilegible ({disease}): {e}")
        return None


def _paths(disease: str):
    real_path = os.path.join(CURATED_DIR, disease, f"{disease}_train.csv")
    synth_files = glob.glob(os.path.join(CURATED_DIR, disease, f"{disease}_synthetic_ctgan*.csv")) \
        or glob.glob(os.path.join(CURATED_DIR, disease, f"{disease}_synthetic*.csv"))
    return real_path, synth_files


def compute(disease: str):
    """Calcula el informe desde los CSV (requiere sdmetrics para el score de
    fidelidad; sin el, devuelve solo las correlaciones)."""
    real_path, synth_files = _paths(disease)
    if not os.path.exists(real_path) or not synth_files:
        return None

    real = pd.read_csv(real_path)
    synth = pd.read_csv(synth_files[0])
    for d in (real, synth):
        if "target" in d.columns:
            d.drop(columns=["target"], inplace=True)
    cols = [c for c in real.columns if c in synth.columns]
    real, synth = real[cols], synth[cols]
    n = min(2000, len(real), len(synth))
    real_s = real.sample(n, random_state=42)
    synth_s = synth.sample(n, random_state=42)

    result = {"n_used": int(n)}

    # --- Score de fidelidad (SDMetrics) ---
    try:
        from sdmetrics.reports.single_table import QualityReport
        meta = {"columns": {c: {"sdtype": "categorical" if real_s[c].nunique() <= 10 else "numerical"} for c in cols}}
        rep = QualityReport()
        rep.generate(real_s, synth_s, meta, verbose=False)
        prop_scores = dict(zip(rep.get_properties()["Property"], rep.get_properties()["Score"]))
        result["overall"] = round(float(rep.get_score()), 4)
        result["column_shapes"] = round(float(prop_scores.get("Column Shapes", 0)), 4)
        result["column_pair_trends"] = round(float(prop_scores.get("Column Pair Trends", 0)), 4)
        try:
            details = rep.get_details("Column Shapes")
            per = [{"column": str(r["Column"]), "score": round(float(r["Score"]), 3)}
                   for _, r in details.iterrows() if pd.notna(r.get("Score"))]
            per.sort(key=lambda x: x["score"], reverse=True)
            result["per_column"] = per
        except Exception as e:
            print(f"quality details fail ({disease}): {e}")
            result["per_column"] = []
    except Exception as e:
        print(f"sdmetrics fail ({disease}): {e}")
        result["overall"] = None
        result["per_column"] = []

    # --- Correlaciones (pandas, robusto) ---
    corr_cols = [c for c in CORR_FEATURES.get(disease, []) if c in cols]
    if len(corr_cols) >= 2:
        rc = real_s[corr_cols].corr().fillna(0).round(2)
        sc = synth_s[corr_cols].corr().fillna(0).round(2)
        result["corr"] = {
            "features": corr_cols,
            "real": rc.values.tolist(),
            "synthetic": sc.values.tolist(),
        }
    return result
