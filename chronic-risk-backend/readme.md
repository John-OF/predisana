# Predisana Backend

API REST en Python/Flask que sirve modelos de Machine Learning para la predicción de riesgo de enfermedades crónicas (diabetes, hipertensión y riesgo cardiovascular). Es el backend de una plataforma educativa: integra inferencia, explicabilidad con SHAP, generación de datos sintéticos con CTGAN/TVAE, una capa de interpretación clínica (ADA / ACC-AHA) y registro de historial en SQLite.

---

## Características

- **3 modelos servidos desde `models/`** (uno por enfermedad). Cada modelo es el **ganador de un bake-off por validación cruzada** (LogReg / RandomForest / LightGBM); no todas las enfermedades usan el mismo algoritmo.
- **Selección de features por respondibilidad** — el simulador es educativo / de autoevaluación, así que cada modelo usa solo features que una persona común puede responder (autorreporte y medición casera/farmacia) y descarta las de laboratorio.
- **Explicabilidad SHAP agnóstica al modelo** — `LinearExplainer` para modelos lineales y `TreeExplainer` para árboles. Cada predicción devuelve el top-5 de variables con mayor impacto.
- **Capa de interpretación clínica desacoplada** — la probabilidad reportada es la salida limpia del modelo; los umbrales diagnósticos (ADA para glucosa/HbA1c, ACC/AHA para presión sistólica) se devuelven aparte como `clinical_flags`/`clinical_note` y **no** modifican la probabilidad.
- **Generación de datos sintéticos** con SDV (CTGAN por defecto, TVAE opcional) para entrenamiento y para la función "caso aleatorio" del simulador.
- **Pipeline de datos por enfermedad** — desde CSVs públicos a un dataset limpio por enfermedad (sin frame maestro concatenado ni imputación cruzada).
- **Historial de predicciones** persistido en SQLite (`medical_history.db`).

---

## Requisitos

- Python 3.12 (fijado en `runtime.txt`, 3.9+ debería funcionar pero no está testeado)
- pip
- Entorno virtual recomendado

---

## Instalación

```powershell
# Crear y activar venv
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# source venv/bin/activate      # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

> Nota: `requirements.txt` lista solo las dependencias **directas** (las transitivas las resuelve pip). Está en UTF-8.

---

## Uso

### Levantar la API

```powershell
python app.py
# Servidor en http://0.0.0.0:8000 (modo debug)
```

Producción (con `gunicorn`, ya incluido en requirements):

```bash
gunicorn app:app
```

Al arrancar, `app.py` ejecuta automáticamente:
1. `_load_all()` — carga los 3 pipelines de `models/`, lee el modelo ganador de cada `_metrics.json` y construye un explainer SHAP acorde al tipo de cada modelo (Linear o Tree).
2. `init_db()` — crea `medical_history.db` y la tabla `predictions` si no existen.

### Pipeline de datos y entrenamiento

Si no tienes los modelos entrenados (o quieres regenerarlos), corre los scripts en orden:

```powershell
# 1. Normaliza los CSV crudos de data_raw/ a un dataset limpio por enfermedad
python prepare_datasets.py

# 2. Split estratificado + síntesis CTGAN/TVAE por enfermedad
python curate_and_synthesize.py

# Opciones útiles
python curate_and_synthesize.py `
    --model ctgan `
    --synth_multiplier 1.0 `
    --balance `
    --epochs 50 `
    --only diabetes,hipertension

# 3. Bake-off multi-modelo por CV: entrena, elige ganador y guarda pkl + métricas + features
python train_models.py
```

### Tests

La suite de `pytest` cubre los invariantes delicados de la API: ruteo del modelo
híbrido de diabetes (con/sin glucosa), alias `glucose`↔`blood_glucose_level`,
monotonía riesgo↔glucosa, calibración isotónica, filtro de género en SHAP, capa
clínica ADA/ACC-AHA, el laboratorio sintético y la auth del panel admin. Corre
contra una base SQLite temporal (nunca toca `medical_history.db`).

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

---

## Endpoints

Todos los endpoints aceptan/devuelven JSON. CORS está habilitado globalmente.

### `GET /health`
Liveness check.

```json
{ "status": "ok", "database": "sqlite_connected" }
```

### `GET /metrics/<disease>`
Devuelve `models/<disease>_metrics.json`. Incluye el modelo ganador (`best_model`), el `leaderboard` con el AUC de CV de cada candidato, y los AUC + classification report de train y test.

### `GET /config/<disease>`
Configuración para construir el formulario en el frontend:

```json
{
  "disease": "diabetes",
  "features": ["age", "bmi", "blood_glucose_level", "..."],
  "ranges": { "age": [18, 100], "bmi": [15, 50], "glucose": [60, 260], "blood_pressure": [60, 130] },
  "categoricals": { "gender": ["Female", "Male"], "smoking_history": ["current", "former", "never", "..."] }
}
```

Las `features` son las propias del esquema de cada enfermedad (heterogéneo), ya recortadas por respondibilidad. Las `categoricals` se derivan de los nombres de las columnas one-hot (`gender_*`, `smoking_history_*`).

### `POST /predict/<disease>`
Recibe un payload con las features clínicas y devuelve la probabilidad de riesgo + explicación SHAP + capa de interpretación clínica.

**Request:**
```json
{
  "age": 55,
  "blood_glucose_level": 165,
  "bmi": 31,
  "hypertension": 1,
  "gender_Male": 1,
  "smoking_history_former": 1
}
```

**Response:**
```json
{
  "disease": "diabetes",
  "model": "lightgbm",
  "probability": 0.91,
  "prediction": 1,
  "missing_filled_as_zero": ["heart_disease"],
  "top_features": [
    { "feature": "blood_glucose_level", "value": 165, "shap": 0.42, "abs_shap": 0.42 },
    { "feature": "bmi", "value": 31, "shap": 0.28, "abs_shap": 0.28 }
  ],
  "clinical_flags": [
    { "indicator": "glucose", "value": 165, "category": "diabetes", "source": "ADA",
      "detail": "Glucosa en ayuno ≥126 mg/dL: criterio de diabetes." }
  ],
  "clinical_note": "Glucosa en ayuno ≥126 mg/dL: criterio de diabetes.",
  "explain_note": "La probabilidad es la salida directa del modelo de ML y los valores SHAP la explican. Los indicadores clínicos (ADA/ACC-AHA) se muestran aparte como referencia y NO modifican la probabilidad."
}
```

**Comportamientos importantes (no triviales):**

- **Aliasing glucosa**: `glucose` y `blood_glucose_level` se espejan automáticamente, así que enviar uno cubre al otro.
- **Capa de interpretación clínica desacoplada (A4)**: `probability` es la salida limpia del modelo (SHAP la explica directamente). Aparte, `clinical_flags`/`clinical_note` exponen los umbrales diagnósticos de referencia (ADA: glucosa ≥100/≥126/≥200, HbA1c ≥5.7/≥6.5; ACC/AHA: sistólica ≥120/≥130/≥140/≥180). Esta capa **no** altera la probabilidad (sustituye al antiguo `max()` con números mágicos).
- **Filtro de género en SHAP**: las features `gender_*` se omiten del top-5 explicativo.
- **Features faltantes**: cualquier feature ausente se rellena con 0; los nombres no-dummy aparecen en `missing_filled_as_zero`.

Cada predicción se persiste en la tabla `predictions` (disease, input_data JSON, prediction, probability, timestamp).

### `GET /synthetic/<disease>`
Devuelve una fila aleatoria de los datos sintéticos curados (`data_curated/<disease>/<disease>_synthetic_ctgan*.csv`). Se usa en el frontend para autocompletar el simulador con un "caso clínico aleatorio". Si no hay sintéticos disponibles, hace fallback a `data_processed/`. El response incluye `_source_type: "synthetic"` o `"real"`.

---

## Estructura

```
chronic-risk-backend/
├── app.py                       # API Flask + carga de modelos + SHAP + capa clínica
├── prepare_datasets.py          # CSVs crudos → un dataset limpio por enfermedad
├── curate_and_synthesize.py     # Split estratificado + síntesis CTGAN/TVAE
├── train_models.py              # Bake-off multi-modelo por CV + persistir ganador
├── requirements.txt             # Dependencias (UTF-16)
├── runtime.txt                  # python-3.12.8
├── medical_history.db           # SQLite generada en runtime (gitignored)
├── data_raw/                    # CSVs públicos originales (Kaggle, ENSANUT, etc.)
├── data_processed/              # Un dataset limpio por enfermedad
├── data_curated/                # Train/test split + sintéticos por enfermedad
└── models/                      # *_pipeline.pkl + *_features.json + *_metrics.json
```

---

## Notas técnicas

- El pipeline sklearn tiene pasos nombrados **`scaler`** y **`clf`**. Estos nombres son load-bearing: SHAP los referencia explícitamente en `_build_shap_explainer`. Renombrarlos rompe la explicabilidad silenciosamente.
- Las 3 enfermedades son un set cerrado declarado en `FILES` (app.py) y en `DATASETS` (train_models.py). Para añadir una nueva, hay que tocar ambos archivos y entrenar el modelo correspondiente.
- `train_models.py` corre un bake-off por enfermedad (LogReg / RandomForest / LightGBM), elige el mejor por AUC en cross-validation y persiste el ganador; el score de cada candidato queda en `_metrics.json.leaderboard`.
- `get_features_for_disease()` parte del esquema propio de cada dataset y solo descarta las features de laboratorio (no respondibles, `DROP_NON_RESPONDABLE`). Con los esquemas por-enfermedad (B1) ya no hay columnas de leakage que recortar.
- `prepare_datasets.py` produce un dataset limpio por enfermedad desde su fuente cruda, con imputación por mediana *intra-dataset* y caps fisiológicos (sin esquema común ni imputación cruzada).
