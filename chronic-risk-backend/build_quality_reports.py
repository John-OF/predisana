# build_quality_reports.py
# Precomputa el informe de calidad del sintetico (SDMetrics + correlaciones) a
# data_curated/<enfermedad>/<enfermedad>_quality.json.
#
# Se corre DESPUES de curate_and_synthesize.py, cada vez que se regenere el
# sintetico. El API sirve estos JSON tal cual, asi que produccion no necesita
# sdmetrics/sdv (que arrastran torch, ~479 MB) — ver AUD-17.
#
#   python build_quality_reports.py                 # las tres enfermedades
#   python build_quality_reports.py --only diabetes
import argparse
import json
import os

import synthetic_quality as sq

DISEASES = ["diabetes", "hipertension", "cardiovascular"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="",
                        help="lista separada por comas (p.ej. diabetes,hipertension)")
    args = parser.parse_args()
    diseases = [d.strip() for d in args.only.split(",") if d.strip()] or DISEASES

    for disease in diseases:
        print(f"\n== {disease} ==")
        res = sq.compute(disease)
        if res is None:
            print("   sin datos curados/sinteticos: se omite")
            continue
        path = sq.report_path(disease)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"   overall={res.get('overall')} n={res.get('n_used')} -> {path}")
        t = res.get("tstr") or {}
        for m in t.get("models", []):
            print(f"   TSTR {m['model']:14s} real {m['trtr_auc']} vs sintetico {m['tstr_auc']} "
                  f"(ratio {m['ratio']})")
        pv = res.get("privacy") or {}
        if pv:
            print(f"   DCR mediana sintetico {pv['median_synthetic']} vs test real "
                  f"{pv['median_real_test']} (ratio {pv['ratio']}) | copias exactas "
                  f"{pv['exact_copies']}/{pv['n_synthetic']} (test real: "
                  f"{pv['exact_copies_real_test']}/{pv['n_real_test']})")


if __name__ == "__main__":
    main()
