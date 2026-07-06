# prepare_nhanes_diabetes.py
# Ingestor NHANES 2021-2023 (ciclo "_L") -> dataset de diabetes para Predisana.
# Une los archivos por SEQN, deriva features RESPONDIBLES (edad, sexo, IMC,
# hipertension y cardiopatia autorreportadas, tabaquismo) + target, y agrega la
# glucemia de laboratorio (glucosa serica + HbA1c) como columnas OPCIONALES para
# el modelo hibrido "con/sin glucosa". Es la fuente canonica de diabetes: escribe
# data_processed/diabetes_dataset.csv (NHANES reemplazo al Kaggle viejo).
import os
import numpy as np
import pandas as pd

DOCS = os.path.join("data_raw", "nhanes")  # archivos .xpt de NHANES
OUT = os.path.join("data_processed", "diabetes_dataset.csv")


def _read(name):
    return pd.read_sas(os.path.join(DOCS, f"{name}.xpt"))


def build():
    demo = _read("DEMO_L")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx = _read("BMX_L")[["SEQN", "BMXBMI"]]
    diq = _read("DIQ_L")[["SEQN", "DIQ010"]]
    bpq = _read("BPQ_L")[["SEQN", "BPQ020"]]
    mcq = _read("MCQ_L")
    smq = _read("SMQ_L")[["SEQN", "SMQ020", "SMQ040"]]
    bio = _read("BIOPRO_L")[["SEQN", "LBXSGL"]]
    ghb = _read("GHB_L")[["SEQN", "LBXGH"]]

    # Base: demografia + target; luego se van uniendo el resto por SEQN.
    df = demo.merge(diq, on="SEQN", how="inner")
    df = df.merge(bmx, on="SEQN", how="left")
    df = df.merge(bpq, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")
    df = df.merge(bio, on="SEQN", how="left")
    df = df.merge(ghb, on="SEQN", how="left")

    # Cardiopatia: cualquiera de las condiciones cardiacas autorreportadas.
    hc_cols = [c for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")
               if c in mcq.columns]
    mcq_hd = mcq[["SEQN"]].copy()
    mcq_hd["heart_disease"] = (mcq[hc_cols] == 1).any(axis=1).astype(int)
    df = df.merge(mcq_hd, on="SEQN", how="left")

    out = pd.DataFrame()
    out["age"] = df["RIDAGEYR"]
    out["bmi"] = df["BMXBMI"]

    # Target: DIQ010 1=Si -> 1 ; 2=No y 3=Borderline -> 0 ; 7/9/NaN -> se descartan.
    tgt = df["DIQ010"]
    out["target"] = np.where(tgt == 1, 1, np.where(tgt.isin([2, 3]), 0, np.nan))

    # Comorbilidades autorreportadas (respondibles). NaN -> 0 (no reportado = no).
    out["hypertension"] = (df["BPQ020"] == 1).astype(int)
    out["heart_disease"] = df["heart_disease"].fillna(0).astype(int)

    # Sexo -> one-hot (RIAGENDR: 1=Masculino, 2=Femenino).
    out["gender_Male"] = (df["RIAGENDR"] == 1).astype(int)
    out["gender_Female"] = (df["RIAGENDR"] == 2).astype(int)

    # Tabaquismo -> categorias respondibles (never/current/former).
    #   SMQ020=2 (nunca 100 cigarrillos)          -> never
    #   SMQ020=1 & SMQ040 in {1,2} (fuma hoy)      -> current
    #   SMQ020=1 & SMQ040=3 (fumo pero ya no)      -> former
    #   resto (NaN/refused)                         -> desconocido (dummies en 0)
    never = df["SMQ020"] == 2
    current = (df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))
    former = (df["SMQ020"] == 1) & (df["SMQ040"] == 3)
    out["smoking_history_never"] = never.astype(int)
    out["smoking_history_current"] = current.astype(int)
    out["smoking_history_former"] = former.astype(int)

    # Glucemia de laboratorio: OPCIONAL (NaN cuando no hay medicion).
    out["blood_glucose_level"] = df["LBXSGL"]
    out["hba1c_level"] = df["LBXGH"]

    # Solo adultos con target y features nucleares (edad, IMC) presentes.
    out = out[(out["age"] >= 18) & out["target"].notna() & out["bmi"].notna()].copy()
    out["target"] = out["target"].astype(int)

    os.makedirs("data_processed", exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


def report(out):
    n = len(out)
    pos = out["target"].mean()
    g = out["blood_glucose_level"]
    h = out["hba1c_level"]
    print(f"\n=== dataset NHANES diabetes -> {OUT} ===")
    print(f"filas: {n}   positivos: {out['target'].sum()} ({pos*100:.1f}%)")
    print(f"columnas: {list(out.columns)}")
    print(f"\nglucosa (LBXSGL): {g.notna().sum()} con dato ({g.notna().mean()*100:.0f}%), "
          f"{g.nunique()} valores unicos, rango {g.min():.0f}-{g.max():.0f}")
    print(f"HbA1c (LBXGH):    {h.notna().sum()} con dato ({h.notna().mean()*100:.0f}%), "
          f"{h.nunique()} valores unicos")
    print(f"\nMODO SIN glucosa (self-report): {n} filas (todas)")
    print(f"MODO CON glucosa: {g.notna().sum()} filas")
    # Tasa de positivos por banda de glucosa (para ver si la senal es suave).
    sub = out[g.notna()]
    print("\ntasa de positivos por banda de glucosa (modo con glucosa):")
    for lo, hi in [(0, 99), (100, 125), (126, 199), (200, 1000)]:
        m = sub[(sub.blood_glucose_level >= lo) & (sub.blood_glucose_level <= hi)]
        if len(m):
            print(f"  {lo:>3}-{hi:<4}: n={len(m):5d}  positivos={m.target.mean()*100:5.1f}%")


if __name__ == "__main__":
    report(build())
