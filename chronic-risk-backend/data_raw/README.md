# data_raw — fuentes crudas (no versionadas)

Los datos crudos **no se versionan** (`data_raw/` está en `.gitignore`, salvo este
README). Son datasets públicos, algunos con licencias de redistribución ambiguas, y
pesan lo suyo; el repo se mantiene liviano y reproducible con los CSV ya procesados
(`data_processed/`) y curados (`data_curated/`), que sí están versionados.

Para **re-correr la ingesta** desde cero, descarga los archivos a las rutas de abajo
y ejecuta `python prepare_datasets.py` (Kaggle) / `python prepare_nhanes_diabetes.py`
(NHANES).

## Diabetes — NHANES 2021-2023 (ciclo "_L")  → `data_raw/nhanes/`

Encuesta real de los CDC (EE. UU.), **dominio público**. Descarga los componentes
`.xpt` del ciclo 2021-2023:
<https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023>

Los usa `prepare_nhanes_diabetes.py` para construir `data_processed/diabetes_dataset.csv`:

- `DEMO_L` (edad/sexo), `BMX_L` (IMC), `DIQ_L` (target diabetes), `BPQ_L`
  (hipertensión autorreportada), `MCQ_L` (cardiopatía), `SMQ_L` (tabaquismo),
  `BIOPRO_L` (glucosa sérica `LBXSGL`), `GHB_L` (HbA1c `LBXGH`).
- `DIQ_L.pdf` es el codebook del cuestionario de diabetes.
- `ALQ_L`, `DBQ_L`, `GLU_L`, `PAQ_L`, `TCHOL_L` están descargados pero **no se usan
  hoy**; se conservan por si se migra el modelo cardiovascular a NHANES.

## Cardiovascular — Kaggle "Cardiovascular Disease dataset"  → `data_raw/cardio_train.csv`

<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

## Diabetes (legado) — Kaggle "Diabetes prediction dataset"  → `data_raw/diabetes_prediction_dataset.csv`

<https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset>

Fue la fuente original de diabetes; **reemplazada por NHANES** (era sintética, con
glucosa cuantizada). Se conserva como referencia histórica del pipeline Kaggle.

## Hipertensión — ENSANUT México  → `data_raw/Hipertension_Arterial_Mexico.csv`

Encuesta Nacional de Salud y Nutrición (México), datos públicos. *(Completar el
enlace exacto de descarga usado.)*
