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

    # Con target: TSTR y DCR lo necesitan. El bloque de fidelidad de mas abajo
    # trabaja sobre copias sin target, que es como se venia calculando.
    real_full, synth_full = real.copy(), synth.copy()
    test_path = os.path.join(CURATED_DIR, disease, f"{disease}_test.csv")
    real_test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    real = real.drop(columns=["target"], errors="ignore")
    synth = synth.drop(columns=["target"], errors="ignore")
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

    # --- Utilidad (TSTR) y privacidad (DCR) ---
    if real_test is not None:
        try:
            result["tstr"] = compute_tstr(real_full, real_test, synth_full)
        except Exception as e:
            print(f"tstr fail ({disease}): {e}")
            result["tstr"] = None
        try:
            result["privacy"] = compute_dcr(real_full, real_test, synth_full)
        except Exception as e:
            print(f"dcr fail ({disease}): {e}")
            result["privacy"] = None

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


# =====================================================================
# UTILIDAD (TSTR) Y PRIVACIDAD (DCR)
# =====================================================================
# El laboratorio enseñaba solo FIDELIDAD (¿se parecen las distribuciones?), que es la
# pregunta facil. Las dos que hace un revisor tecnico son:
#   - TSTR (Train on Synthetic, Test on Real): ¿SIRVE el sintetico? Se entrena un
#     modelo SOLO con datos sinteticos y se evalua contra datos reales que nunca vio
#     nadie. Se compara con TRTR (entrenado con reales) sobre EL MISMO test.
#   - DCR (Distance to Closest Record): ¿el GAN MEMORIZO? Si las filas sinteticas
#     estan mas pegadas a los datos reales de lo que lo estan dos muestras reales
#     entre si, el sintetico es una copia disfrazada y deja de ser anonimo.
# Ambas necesitan solo sklearn/numpy/pandas, que ya son dependencias del API; aun asi
# se precomputan como el resto del informe (AUD-17) para no entrenar en cada request.
TSTR_SEED = 42


def _arreglar_columnas(real_tr, real_te, synth):
    """Columnas comunes a los tres conjuntos, con target aparte."""
    cols = [c for c in real_tr.columns
            if c in real_te.columns and c in synth.columns and c != "target"]
    return cols


def _entrenar_y_evaluar(X_tr, y_tr, X_te, y_te):
    """AUC en el test REAL de dos modelos fijos. Se usan los MISMOS para las dos
    ramas: asi la comparacion aisla los datos, no el algoritmo."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    modelos = {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=TSTR_SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                           random_state=TSTR_SEED, n_jobs=-1)),
        ]),
    }
    out = {}
    for nombre, pipe in modelos.items():
        if len(set(y_tr)) < 2:      # una sola clase: el AUC no esta definido
            out[nombre] = None
            continue
        pipe.fit(X_tr, y_tr)
        out[nombre] = round(float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])), 4)
    return out


def compute_tstr(real_tr, real_te, synth):
    """TSTR vs TRTR sobre el mismo test real. El ratio es la lectura corta: 1.0 seria
    'el sintetico sirve tanto como el real para entrenar'."""
    if "target" not in synth.columns or "target" not in real_tr.columns:
        return None
    cols = _arreglar_columnas(real_tr, real_te, synth)
    if not cols:
        return None

    te = real_te.dropna(subset=["target"])
    X_te, y_te = te[cols], te["target"].astype(int)
    if y_te.nunique() < 2:
        return None

    tr = real_tr.dropna(subset=["target"])
    sy = synth.dropna(subset=["target"])
    trtr = _entrenar_y_evaluar(tr[cols], tr["target"].astype(int), X_te, y_te)
    tstr = _entrenar_y_evaluar(sy[cols], sy["target"].astype(int), X_te, y_te)

    modelos = []
    for nombre in trtr:
        a, b = trtr.get(nombre), tstr.get(nombre)
        modelos.append({
            "model": nombre,
            "trtr_auc": a,
            "tstr_auc": b,
            "ratio": round(b / a, 3) if a and b else None,
        })
    return {
        "n_train_real": int(len(tr)), "n_train_synth": int(len(sy)),
        "n_test_real": int(len(te)), "features": cols, "models": modelos,
    }


def compute_dcr(real_tr, real_te, synth):
    """Distancia al vecino real mas cercano. La referencia honesta no es cero, es el
    propio TEST real: dos muestras distintas de la misma poblacion tambien se parecen.
    Si el sintetico queda MAS LEJOS que ese baseline, no hay memorizacion."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    cols = _arreglar_columnas(real_tr, real_te, synth)
    if not cols:
        return None

    imp = SimpleImputer(strategy="median").fit(real_tr[cols])
    esc = StandardScaler().fit(imp.transform(real_tr[cols]))
    prep = lambda d: esc.transform(imp.transform(d[cols]))

    base = prep(real_tr)
    nn = NearestNeighbors(n_neighbors=1).fit(base)
    d_synth = nn.kneighbors(prep(synth))[0].ravel()
    d_test = nn.kneighbors(prep(real_te))[0].ravel()

    # Copias exactas: el caso que de verdad rompe el anonimato. Se mide TAMBIEN entre
    # el test real y el train real: en datos gruesos (cardiovascular son enteros) dos
    # personas distintas comparten fila a menudo, y sin esa referencia un "4 copias"
    # se lee como alarma cuando es ruido del dominio.
    llaves = lambda d: set(map(tuple, d[cols].round(4).itertuples(index=False, name=None)))
    k_train = llaves(real_tr)
    copias = len(llaves(synth) & k_train)
    copias_test = len(llaves(real_te) & k_train)

    med_s, med_t = float(np.median(d_synth)), float(np.median(d_test))
    return {
        "median_synthetic": round(med_s, 4),
        "median_real_test": round(med_t, 4),
        "ratio": round(med_s / med_t, 3) if med_t else None,
        "share_near_zero_synthetic": round(float((d_synth < 0.01).mean()), 4),
        "share_near_zero_real_test": round(float((d_test < 0.01).mean()), 4),
        "exact_copies": int(copias),
        "exact_copies_real_test": int(copias_test),
        "n_synthetic": int(len(synth)),
        "n_real_test": int(len(real_te)),
    }
