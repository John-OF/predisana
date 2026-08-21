# prepare_nhanes_hipertension.py
# Ingestor NHANES 2021-2023 (ciclo "_L") -> dataset de hipertension para Predisana.
#
# POR QUE EXISTE (AUD-1): el dataset anterior venia de un CSV de ENSANUT cuyo target
# `riesgo_hipertension` NO era un desenlace clinico sino una formula del autor. El
# modelo la reaprendia y salian relaciones invertidas (a mas edad, MENOS riesgo; un
# joven de 25 con presion 110 recibia 100%). Ademas la presion medida casi no
# correlacionaba con ese target (r=0.059) y el 70% de los labs eran la mediana
# imputada. Se reemplaza por NHANES, igual que se hizo con diabetes.
#
# Target: BPQ020 ("¿alguna vez le dijeron que tenia la presion alta?"), autorreporte
# real. Se EXCLUYEN a proposito BPQ030 (se lo dijeron 2+ veces) y BPQ150 (toma
# medicacion): son consecuencia del diagnostico, no factores de riesgo -> leakage.
#
# Features: solo RESPONDIBLES (misma politica que diabetes). No hay presion medida
# en el ciclo descargado (falta BPXO_L) y tampoco se busca: meter la tension como
# feature convertiria el modelo en un umbral disfrazado. La presion, si el usuario
# la conoce, se usa aparte en la capa clinica ACC/AHA.
import os

import numpy as np
import pandas as pd

RAW = os.path.join("data_raw", "nhanes")
OUT = os.path.join("data_processed", "hipertension_dataset.csv")


def _read(name):
    return pd.read_sas(os.path.join(RAW, f"{name}.xpt"))


def build():
    demo = _read("DEMO_L")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bpq = _read("BPQ_L")[["SEQN", "BPQ020", "BPQ080"]]
    bmx = _read("BMX_L")[["SEQN", "BMXBMI", "BMXWT", "BMXWAIST"]]
    diq = _read("DIQ_L")[["SEQN", "DIQ010"]]
    smq = _read("SMQ_L")[["SEQN", "SMQ020", "SMQ040"]]
    mcq = _read("MCQ_L")

    df = demo.merge(bpq, on="SEQN", how="inner")
    df = df.merge(bmx, on="SEQN", how="left")
    df = df.merge(diq, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")

    # Cardiopatia: cualquiera de las condiciones cardiacas autorreportadas.
    hc_cols = [c for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")
               if c in mcq.columns]
    mcq_hd = mcq[["SEQN"]].copy()
    mcq_hd["heart_disease"] = (mcq[hc_cols] == 1).any(axis=1).astype(int)
    df = df.merge(mcq_hd, on="SEQN", how="left")

    out = pd.DataFrame()
    out["age"] = df["RIDAGEYR"]
    out["bmi"] = df["BMXBMI"]
    out["weight"] = df["BMXWT"]
    out["waist_circumference"] = df["BMXWAIST"]

    # Target: BPQ020 1=Si -> 1 ; 2=No -> 0 ; 7/9/NaN -> se descartan.
    tgt = df["BPQ020"]
    out["target"] = np.where(tgt == 1, 1, np.where(tgt == 2, 0, np.nan))

    # Comorbilidades autorreportadas (respondibles, no laboratorio).
    out["diabetes"] = (df["DIQ010"] == 1).astype(int)
    out["heart_disease"] = df["heart_disease"].fillna(0).astype(int)
    out["high_cholesterol"] = (df["BPQ080"] == 1).astype(int)

    # Sexo -> one-hot (RIAGENDR: 1=Masculino, 2=Femenino).
    out["gender_Male"] = (df["RIAGENDR"] == 1).astype(int)
    out["gender_Female"] = (df["RIAGENDR"] == 2).astype(int)

    # Tabaquismo -> never/current/former (mismo criterio que en diabetes).
    never = df["SMQ020"] == 2
    current = (df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))
    former = (df["SMQ020"] == 1) & (df["SMQ040"] == 3)
    out["smoking_history_never"] = never.astype(int)
    out["smoking_history_current"] = current.astype(int)
    out["smoking_history_former"] = former.astype(int)

    # Solo adultos con target y antropometria presentes (sin imputar nada).
    out = out[(out["age"] >= 18) & out["target"].notna()
              & out["bmi"].notna() & out["waist_circumference"].notna()].copy()
    out["target"] = out["target"].astype(int)
    out = out.round(2).reset_index(drop=True)

    os.makedirs("data_processed", exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


def report(out):
    print(f"\n=== dataset NHANES hipertension -> {OUT} ===")
    print(f"filas: {len(out)}   positivos: {out['target'].sum()} ({out['target'].mean()*100:.1f}%)")
    print(f"columnas: {list(out.columns)}")
    print(f"NaN restantes: {int(out.isna().sum().sum())}")

    print("\nprevalencia por decada de edad (debe CRECER):")
    for lo in range(18, 90, 10):
        m = out[(out.age >= lo) & (out.age < lo + 10)]
        if len(m):
            print(f"  {lo}-{lo+9}: n={len(m):5d}  {m.target.mean()*100:5.1f}%")

    print("\nprevalencia por banda de IMC (debe CRECER):")
    for lo, hi in [(0, 25), (25, 30), (30, 35), (35, 100)]:
        m = out[(out.bmi >= lo) & (out.bmi < hi)]
        if len(m):
            print(f"  IMC {lo:>2}-{hi:<3}: n={len(m):5d}  {m.target.mean()*100:5.1f}%")

    print("\ncomorbilidades autorreportadas:")
    for c in ("diabetes", "heart_disease", "high_cholesterol"):
        si, no = out[out[c] == 1], out[out[c] == 0]
        print(f"  {c:17s} si: n={len(si):5d} {si.target.mean()*100:5.1f}%  |  "
              f"no: n={len(no):5d} {no.target.mean()*100:5.1f}%")


if __name__ == "__main__":
    report(build())
