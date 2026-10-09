# data_raw — fuentes crudas (no versionadas)

Los datos crudos **no se versionan** (`data_raw/` está en `.gitignore`, salvo este
README). Pesan lo suyo y el repo se mantiene liviano y reproducible con los CSV ya
procesados (`data_processed/`) y curados (`data_curated/`), que sí están versionados.

Para **re-correr la ingesta** desde cero, descarga los archivos a `data_raw/nhanes/` y
ejecuta `python prepare_nhanes.py` (escribe `data_processed/`).

## NHANES 2017-2020 + 2021-2023 → `data_raw/nhanes/`

Desde la v2 todas las enfermedades salen de **NHANES**, la encuesta de salud y nutrición
de los CDC (EE. UU.), con los dos ciclos juntos: 2017-marzo 2020 prepandemia (archivos
`P_*`) y 2021-2023 (`*_L`). Todo es **dominio público** y se descarga sin cuenta:

```bash
cd data_raw/nhanes
for c in DEMO BMX BPXO BPQ DIQ MCQ SMQ WHQ GHB GLU BIOPRO TCHOL HDL ALB_CR KIQ_U LUX; do
  curl -sLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/${c}_L.xpt"
  curl -sLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_${c}.xpt"
done
```

Qué sale de cada uno: `DEMO` (edad, sexo, embarazo, pesos muestrales), `BMX` (peso,
talla, IMC y cintura medidos), `WHQ` (peso y talla autodeclarados, los del modo
simplificado), `BPXO` (presión medida, 3 lecturas), `BPQ` (hipertensión y colesterol
alto diagnosticados, medicación), `DIQ` (diabetes diagnosticada), `MCQ` (cardiopatías e
ictus), `SMQ` (tabaco), `GHB` (HbA1c), `GLU` (glucosa en ayunas, submuestra), `BIOPRO`
(creatinina para el eGFR), `TCHOL` y `HDL` (colesterol), `ALB_CR` (albúmina/creatinina
en orina), `KIQ_U` (enfermedad renal diagnosticada: "riñones débiles o en fallo"), `LUX` (elastografía FibroScan: el CAP que define el hígado graso; la ALT sale de `BIOPRO`).

Se descargaron también y **no se usan**: `PAQ` (actividad física: el cuestionario
cambió entre ciclos y no hay forma honesta de igualarlo, ver la cabecera de
`prepare_nhanes.py`), `ALQ` (alcohol: en datos de un solo momento quien enferma deja de
beber y sale "protector") y `TRIGLY` (triglicéridos, solo en la submuestra en ayunas).

## Fuentes de la v1 (retiradas)

- **Cardiovascular — Kaggle "Cardiovascular Disease dataset"**
  (<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>). Sus
  hábitos eran autorreportados y ahí los fumadores enfermaban menos: el modelo
  aprendía que fumar protege. Su edad iba de 29,7 a 64,9 años.
- **Diabetes — Kaggle "Diabetes prediction dataset"**
  (<https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset>). Era
  sintética, con glucosa cuantizada. La v1 ya la había cambiado por NHANES 2021-2023.
- **Hipertensión — ENSANUT México** (`Hipertension_Arterial_Mexico.csv`). Su columna
  `riesgo_hipertension` no era un desenlace clínico sino una fórmula del autor del CSV:
  el modelo la reaprendía y salían relaciones invertidas (la presión medida apenas
  correlacionaba con ella, r=0,06, y ~70% de los labs eran la mediana imputada).
