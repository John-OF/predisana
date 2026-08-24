# data_raw — fuentes crudas (no versionadas)

Los datos crudos **no se versionan** (`data_raw/` está en `.gitignore`, salvo este
README). Son datasets públicos, algunos con licencias de redistribución ambiguas, y
pesan lo suyo; el repo se mantiene liviano y reproducible con los CSV ya procesados
(`data_processed/`) y curados (`data_curated/`), que sí están versionados.

Para **re-correr la ingesta** desde cero, descarga los archivos a las rutas de abajo
y ejecuta `python prepare_datasets.py` (Kaggle, cardiovascular),
`python prepare_nhanes_diabetes.py` y `python prepare_nhanes_hipertension.py` (NHANES).

## Diabetes — NHANES 2021-2023 (ciclo "_L")  → `data_raw/nhanes/`

Encuesta real de los CDC (EE. UU.), **dominio público**. Descarga los componentes
`.xpt` del ciclo 2021-2023:
<https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023>

Los usa `prepare_nhanes_diabetes.py` para construir `data_processed/diabetes_dataset.csv`:

Fuente de **dos** datasets: diabetes (`prepare_nhanes_diabetes.py`) e
**hipertensión** (`prepare_nhanes_hipertension.py`).

- `DEMO_L` (edad/sexo), `BMX_L` (IMC, peso, cintura), `SMQ_L` (tabaquismo),
  `MCQ_L` (cardiopatía autorreportada).
- Diabetes: `DIQ_L` (target `DIQ010`), `BIOPRO_L` (glucosa sérica `LBXSGL`),
  `GHB_L` (HbA1c `LBXGH`), `BPQ_L` (hipertensión autorreportada, como feature).
- Hipertensión: `BPQ_L` (target `BPQ020` + `BPQ080` colesterol alto autorreportado),
  `DIQ_L` (diabetes autorreportada, como feature). Se excluyen a propósito `BPQ030`
  y `BPQ150` (medicación): son consecuencia del diagnóstico, no factores de riesgo.
- `DIQ_L.pdf` es el codebook del cuestionario de diabetes.
- `ALQ_L`, `DBQ_L`, `GLU_L`, `PAQ_L`, `TCHOL_L` están descargados pero **no se usan
  hoy**; se conservan por si se migra el modelo cardiovascular a NHANES.
- No está `BPXO_L` (presión medida) y no hace falta: la tensión no entra al modelo
  de hipertensión (sería un umbral disfrazado), solo se interpreta en la capa
  clínica ACC/AHA si el usuario la conoce.

## Cardiovascular — Kaggle "Cardiovascular Disease dataset"  → `data_raw/cardio_train.csv`

<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

## Diabetes (legado) — Kaggle "Diabetes prediction dataset"  → `data_raw/diabetes_prediction_dataset.csv`

<https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset>

Fue la fuente original de diabetes; **reemplazada por NHANES** (era sintética, con
glucosa cuantizada). Se conserva como referencia histórica del pipeline Kaggle.

## Hipertensión (legado) — ENSANUT México  → `data_raw/Hipertension_Arterial_Mexico.csv`

Encuesta Nacional de Salud y Nutrición (México), datos públicos. **Reemplazada por
NHANES**: su columna `riesgo_hipertension` no era un desenlace clínico sino una
fórmula del autor del CSV (el modelo la reaprendía y salían relaciones invertidas;
la presión medida apenas correlacionaba con ella, r=0.06, y ~70% de los labs eran
la mediana imputada). Se conserva como referencia histórica.
