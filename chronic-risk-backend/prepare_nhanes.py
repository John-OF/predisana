# prepare_nhanes.py
# Ingesta NHANES v2: dos ciclos (2017-2020 prepandemia, "P_", y 2021-2023, "_L"), las
# enfermedades y las variables de los dos modos (simplificado y completo).
# Sustituye a prepare_nhanes_diabetes.py, prepare_nhanes_hipertension.py y al CSV de
# Kaggle de cardiovascular (prepare_datasets.py). Escribe data_processed/.
#
# Lo que cambia respecto a la v1 (auditoria de datos, revision 2026-10):
# - Dos ciclos en vez de uno: de ~6.000 adultos por enfermedad a 13.000-16.000.
# - Objetivo = enfermedad TOTAL, no "alguna vez se lo dijeron": segun el ciclo, el
#   22-24% de quienes tienen diabetes y el 15-17% de quienes tienen hipertension no
#   estaban diagnosticados.
#     diabetes:      diagnosticada, HbA1c >= 6,5% o glucosa en ayunas >= 126 mg/dL (ADA)
#     hipertension:  diagnosticada, >= 140/90 mmHg medida o medicacion (OMS/ESC)
#     cardiovascular: cardiopatia coronaria, angina, infarto, insuficiencia cardiaca o
#                     ictus autorreportados (no hay definicion por analitica)
#     renal:         diagnosticada (rinones debiles o en fallo), filtrado glomerular
#                    (eGFR) < 60 o albumina/creatinina en orina >= 30 mg/g (KDIGO). Es
#                    una sola medicion, como en la vigilancia de los CDC: el diagnostico
#                    clinico exige que dure mas de 3 meses. El 81% de quienes la tienen
#                    no estaba diagnosticado.
#     higado:        esteatosis hepatica por elastografia (FibroScan): CAP >= 288 dB/m
#                    en una exploracion valida. Solo la miden las personas de 12 a 79
#                    anos que se hicieron la exploracion; el objetivo es medido, no
#                    autodeclarado. No separa causas (alcohol, virus): es "higado graso".
#   Solo entra quien tiene la medicion que define el objetivo (HbA1c, presion; en
#   renal, la creatinina y la orina): si no, los no diagnosticados sin analitica
#   contarian como sanos aunque no lo sepamos.
# - La glucosa es la de AYUNAS (LBXGLU). La v1 usaba la del perfil bioquimico
#   (LBXSGL), con y sin ayuno mezclados, y la app la leia con umbrales de ayunas.
# - El modo simplificado usa el peso y la talla AUTODECLARADOS (WHQ): es lo que el
#   usuario va a escribir, y la gente se quita IMC (-0,8 de media, -1,5 con obesidad y
#   -2,3 con un IMC de 40 o mas).
# - Fuera las embarazadas (peso, IMC y presion no comparables), y los "no sabe" /
#   "se niega" son dato faltante, no un "no".
# - Sin actividad fisica: el cuestionario cambio entre ciclos y no hay forma honesta
#   de igualarlo. Cumplir los 150 min/semana de la OMS daba 34% en 2017-2020 y 53% en
#   2021-2023; "alguna actividad en el tiempo libre", 48% y 80% (el nuevo cuenta hasta
#   lo que se hace una vez al ano). Con la variable dentro, el modelo aprenderia el
#   cuestionario, no el ejercicio. Tampoco alcohol: en datos de un solo momento quien
#   enferma deja de beber y sale "protector", como paso con el tabaco en Kaggle.
import os

import numpy as np
import pandas as pd

RAW = os.path.join("data_raw", "nhanes")
OUT_DIR = "data_processed"

# Duracion de cada ciclo: al combinarlos, el peso muestral se reparte en proporcion.
CICLOS = {
    "2017-2020": {"nombre": lambda c: f"P_{c}", "anios": 3.2, "peso_entrevista": "WTINTPRP",
                  "peso_examen": "WTMECPRP", "medicacion_hta": "BPQ050A"},
    "2021-2023": {"nombre": lambda c: f"{c}_L", "anios": 2.0, "peso_entrevista": "WTINT2YR",
                  "peso_examen": "WTMEC2YR", "medicacion_hta": "BPQ150"},
}
COMPONENTES = ("DEMO", "BMX", "BPXO", "BPQ", "DIQ", "MCQ", "SMQ", "WHQ", "GHB", "GLU",
               "BIOPRO", "TCHOL", "HDL", "ALB_CR", "KIQ_U", "LUX")
CARDIO = ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")  # IC, coronaria, angina, infarto, ictus


def _leer(ciclo, comp):
    df = pd.read_sas(os.path.join(RAW, CICLOS[ciclo]["nombre"](comp) + ".xpt"), encoding="latin-1")
    # El formato XPT guarda el 0 como 5.4e-79: se devuelve a 0.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    return df


def _si_no(serie):
    """1 = si, 2 = no; 7 (se niega), 9 (no sabe) y vacio = desconocido."""
    return serie.map({1: 1.0, 2: 0.0})


def _sin_codigos(serie, *codigos):
    return serie.where(~serie.isin(codigos))


def _egfr(creatinina, edad, mujer):
    """CKD-EPI 2021, sin raza (mL/min/1,73 m2)."""
    k = np.where(mujer, 0.7, 0.9)
    alfa = np.where(mujer, -0.241, -0.302)
    r = creatinina / k
    return (142 * np.minimum(r, 1) ** alfa * np.maximum(r, 1) ** -1.2
            * 0.9938 ** edad * np.where(mujer, 1.012, 1.0))


def _ciclo(ciclo):
    cfg = CICLOS[ciclo]
    df = _leer(ciclo, "DEMO")
    for comp in COMPONENTES[1:]:
        extra = _leer(ciclo, comp)
        # Varios laboratorios repiten columnas (el peso de la extraccion): la primera vale.
        extra = extra[["SEQN"] + [c for c in extra.columns if c not in df.columns]]
        df = df.merge(extra, on="SEQN", how="left", validate="one_to_one")

    df = df[(df["RIDAGEYR"] >= 18) & (df["RIDEXPRG"] != 1)].copy()
    out = pd.DataFrame(index=df.index)
    out["ciclo"] = ciclo
    # Dos pesos: el de entrevista vale para todos; el de examen es 0 para quien solo
    # hizo la entrevista (en cardiovascular, el objetivo sale de la entrevista).
    reparto = cfg["anios"] / sum(c["anios"] for c in CICLOS.values())
    out["peso_entrevista"] = df[cfg["peso_entrevista"]] * reparto
    out["peso_examen"] = df[cfg["peso_examen"]] * reparto

    # ---- Comunes a los dos modos
    out["age"] = df["RIDAGEYR"]   # 80 = "80 o mas" (tope de NHANES)
    out["gender_Male"] = (df["RIAGENDR"] == 1).astype(int)
    out["gender_Female"] = (df["RIAGENDR"] == 2).astype(int)
    fuma = df["SMQ020"].map({1: 1, 2: 0})
    hoy = df["SMQ040"]
    out["smoking_history_never"] = np.where(fuma.isna(), np.nan, (fuma == 0).astype(float))
    out["smoking_history_current"] = np.where(fuma.isna(), np.nan, ((fuma == 1) & hoy.isin([1, 2])).astype(float))
    out["smoking_history_former"] = np.where(fuma.isna(), np.nan, ((fuma == 1) & (hoy == 3)).astype(float))
    # Fumo alguna vez pero no consta si fuma hoy: categoria desconocida.
    sin_dato = (fuma == 1) & ~hoy.isin([1, 2, 3])
    for c in ("smoking_history_never", "smoking_history_current", "smoking_history_former"):
        out.loc[sin_dato, c] = np.nan
    out["diabetes"] = df["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0})   # 3 = "borderline": no diagnosticada
    out["hypertension"] = _si_no(df["BPQ020"])
    out["high_cholesterol"] = _si_no(df["BPQ080"])
    cardio = df[list(CARDIO)]
    hd = np.where((cardio == 1).any(axis=1), 1.0, np.where((cardio == 2).all(axis=1), 0.0, np.nan))
    # Se pregunta desde los 20 anos: a los de 18-19 les falta por diseno.
    out["heart_disease"] = np.where(cardio.isna().all(axis=1) & (df["RIDAGEYR"] < 20), 0.0, hd)

    # ---- Simplificado: peso y talla autodeclarados (pulgadas y libras)
    talla = _sin_codigos(df["WHD010"], 7777, 9999) * 2.54
    peso = _sin_codigos(df["WHD020"], 7777, 9999) * 0.45359237
    imc = peso / (talla / 100) ** 2
    plausible = talla.between(120, 230) & peso.between(30, 300) & imc.between(12, 90)
    out["weight"] = peso.where(plausible).round(1)       # lo que escribe el usuario
    out["height"] = talla.where(plausible).round(1)
    out["bmi_autodeclarado"] = imc.where(plausible).round(1)

    # ---- Completo: lo que mide el personal sanitario
    out["bmi"] = df["BMXBMI"]
    out["waist_circumference"] = df["BMXWAIST"]
    out["ap_hi"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1).round(1)
    out["ap_lo"] = df[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1).round(1)
    out["total_cholesterol"] = df["LBXTC"]
    out["hdl_cholesterol"] = df["LBDHDD"]
    out["hba1c_level"] = df["LBXGH"]
    out["blood_glucose_level"] = df["LBXGLU"]          # en ayunas (submuestra)
    out["egfr"] = _egfr(df["LBXSCR"], df["RIDAGEYR"], df["RIAGENDR"] == 2).round(1)
    out["albumin_creatinine_ratio"] = df["URDACT"].round(2)
    out["alt"] = df["LBXSATSI"]                         # transaminasa ALT (U/L)

    # ---- Objetivos (enfermedad total); NaN = no determinable
    medicado = _si_no(df[cfg["medicacion_hta"]]).fillna(0)  # solo se pregunta a diagnosticados
    alta = (out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)
    out["obj_hipertension"] = np.where(
        out["hypertension"].isna() | out["ap_hi"].isna(), np.nan,
        ((out["hypertension"] == 1) | alta | (medicado == 1)).astype(float))
    lab = (out["hba1c_level"] >= 6.5) | (out["blood_glucose_level"] >= 126)
    out["obj_diabetes"] = np.where(
        out["diabetes"].isna() | out["hba1c_level"].isna(), np.nan,
        ((out["diabetes"] == 1) | lab).astype(float))
    out["obj_cardiovascular"] = np.where(df["RIDAGEYR"] >= 20, out["heart_disease"], np.nan)
    # "Rinones debiles o en fallo" (sin calculos, infecciones ni incontinencia). Como las
    # cardiopatias, se pregunta desde los 20: a los de 18-19 solo los define la analitica.
    renal_dx = _si_no(df["KIQ022"])
    renal_dx = renal_dx.where(~(renal_dx.isna() & (df["RIDAGEYR"] < 20)), 0.0)
    renal_lab = (out["egfr"] < 60) | (out["albumin_creatinine_ratio"] >= 30)
    out["obj_renal"] = np.where(
        renal_dx.isna() | out["egfr"].isna() | out["albumin_creatinine_ratio"].isna(), np.nan,
        ((renal_dx == 1) | renal_lab).astype(float))
    # Higado graso: solo cuenta la exploracion valida (LUAXSTAT 1) con CAP medido.
    valida = (df["LUAXSTAT"] == 1) & df["LUXCAPM"].notna()
    out["obj_higado"] = np.where(valida, (df["LUXCAPM"] >= 288).astype(float), np.nan)
    return out


# Columnas de cada dataset. Las "definitorias" (las que definen el objetivo) se
# guardan para la capa clinica y el laboratorio, pero ningun modelo las usa.
DATASETS = {
    "diabetes": ["age", "gender_Male", "gender_Female", "smoking_history_never",
                 "smoking_history_current", "smoking_history_former",
                 "hypertension", "high_cholesterol", "heart_disease", "weight", "height", "bmi_autodeclarado",
                 "bmi", "waist_circumference", "ap_hi", "ap_lo", "total_cholesterol",
                 "hdl_cholesterol", "egfr", "albumin_creatinine_ratio",
                 "hba1c_level", "blood_glucose_level"],
    "hipertension": ["age", "gender_Male", "gender_Female", "smoking_history_never",
                     "smoking_history_current", "smoking_history_former",
                     "diabetes", "high_cholesterol", "heart_disease", "weight", "height", "bmi_autodeclarado",
                     "bmi", "waist_circumference", "total_cholesterol", "hdl_cholesterol",
                     "hba1c_level", "egfr", "albumin_creatinine_ratio", "ap_hi", "ap_lo"],
    "cardiovascular": ["age", "gender_Male", "gender_Female", "smoking_history_never",
                       "smoking_history_current", "smoking_history_former",
                       "diabetes", "hypertension", "high_cholesterol", "weight", "height",
                       "bmi_autodeclarado",
                       "bmi", "waist_circumference", "ap_hi", "ap_lo", "total_cholesterol",
                       "hdl_cholesterol", "hba1c_level", "egfr", "albumin_creatinine_ratio"],
    "renal": ["age", "gender_Male", "gender_Female", "smoking_history_never",
              "smoking_history_current", "smoking_history_former",
              "diabetes", "hypertension", "high_cholesterol", "heart_disease", "weight", "height",
              "bmi_autodeclarado",
              "bmi", "waist_circumference", "ap_hi", "ap_lo", "total_cholesterol",
              "hdl_cholesterol", "hba1c_level", "egfr", "albumin_creatinine_ratio"],
    "higado": ["age", "gender_Male", "gender_Female", "smoking_history_never",
               "smoking_history_current", "smoking_history_former",
               "diabetes", "hypertension", "high_cholesterol", "heart_disease", "weight", "height",
               "bmi_autodeclarado",
               "bmi", "waist_circumference", "ap_hi", "ap_lo", "total_cholesterol",
               "hdl_cholesterol", "hba1c_level", "alt"],
}


def build():
    todo = pd.concat([_ciclo(c) for c in CICLOS], ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    salida = {}
    for nombre, columnas in DATASETS.items():
        d = todo[todo[f"obj_{nombre}"].notna()]
        d = d[["ciclo", "peso_entrevista", "peso_examen"] + columnas].assign(
            target=d[f"obj_{nombre}"].astype(int)).reset_index(drop=True)
        d.to_csv(os.path.join(OUT_DIR, f"{nombre}_dataset.csv"), index=False)
        salida[nombre] = d
    return salida


def report(salida):
    for nombre, d in salida.items():
        print(f"\n=== {nombre}: {len(d)} adultos, {d['target'].mean():.1%} con la enfermedad ===")
        print("   por ciclo:", d.groupby("ciclo")["target"].agg(["size", "mean"]).round(3).to_dict("index"))
        print("   por edad:", {f"{lo}-{lo + 19}": round(d[d.age.between(lo, lo + 19)].target.mean(), 3)
                               for lo in (18, 38, 58, 78) if (d.age.between(lo, lo + 19)).any()})
        faltan = d.drop(columns=["ciclo", "peso_entrevista", "peso_examen", "target"]).isna().mean()
        print("   faltantes:", {k: f"{v:.0%}" for k, v in faltan.items() if v > 0})


if __name__ == "__main__":
    report(build())
