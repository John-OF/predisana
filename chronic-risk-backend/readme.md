# Predisana Backend

API REST en Python/Flask que sirve modelos de Machine Learning para la predicción de riesgo de enfermedades crónicas (diabetes, hipertensión, obesidad y riesgo cardiovascular). Es el backend de una plataforma educativa: integra inferencia, explicabilidad con SHAP, generación de datos sintéticos con CTGAN/TVAE, reglas clínicas progresivas y registro de historial en SQLite.

---

## Características

- **4 modelos de Regresión Logística** servidos desde `models/` (uno por enfermedad).
- **Explicabilidad SHAP** — cada predicción devuelve el top-5 de variables con mayor impacto.
- **Reglas clínicas progresivas** sobre la probabilidad del modelo (umbrales de glucosa, HbA1c y presión arterial). Para obesidad se aplica el criterio OMS (BMI ≥ 30) cuando hay BMI disponible.
- **Generación de datos sintéticos** con SDV (CTGAN por defecto, TVAE opcional) para entrenamiento y para la función "caso aleatorio" del simulador.
- **Pipeline completo de datos** desde CSVs públicos heterogéneos a datasets curados listos para entrenar.
- **Historial de predicciones** persistido en SQLite (`medical_history.db`).
- **Diccionario de datos** generado automáticamente en CSV y Markdown por dataset.

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

> Nota: `requirements.txt` está codificado en UTF-16. Si necesitas editarlo, respeta la codificación o reescríbelo en UTF-8.

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
1. `_load_all()` — carga los 4 pipelines de `models/` y construye un `LinearExplainer` de SHAP por modelo.
2. `init_db()` — crea `medical_history.db` y la tabla `predictions` si no existen.

### Pipeline de datos y entrenamiento

Si no tienes los modelos entrenados (o quieres regenerarlos), corre los scripts en orden:

```powershell
# 1. Normaliza los CSV crudos de data_raw/ a un esquema común
python prepare_datasets.py

# 2. Split estratificado + síntesis CTGAN/TVAE + diccionario de datos
python curate_and_synthesize.py

# Opciones útiles
python curate_and_synthesize.py `
    --model ctgan `
    --synth_multiplier 1.0 `
    --balance `
    --epochs 50 `
    --only diabetes,hipertension

# 3. Entrena los 4 modelos y guarda pkl + métricas + features
python train_models.py
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
Devuelve `models/<disease>_metrics.json` (incluye AUC y classification report tanto de train como de test).

### `GET /config/<disease>`
Configuración para construir el formulario en el frontend:

```json
{
  "disease": "diabetes",
  "features": ["age", "glucose", "bmi", "..."],
  "ranges": { "age": [18, 100], "bmi": [15, 50], "glucose": [60, 260], "blood_pressure": [60, 130] },
  "categoricals": { "gender": ["Female", "Male"], "smoking_history": ["current", "former", "never", "..."] }
}
```

### `POST /predict/<disease>`
Recibe un payload con las features clínicas y devuelve la probabilidad de riesgo + explicación SHAP.

**Request:**
```json
{
  "age": 55,
  "glucose": 165,
  "bmi": 31,
  "hba1c_level": 7.2,
  "blood_pressure": 140,
  "gender_Male": 1,
  "smoking_history_former": 1
}
```

**Response:**
```json
{
  "disease": "diabetes",
  "probability": 0.91,
  "prediction": 1,
  "raw_model_probability": 0.78,
  "missing_filled_as_zero": ["insulin", "skin_thickness"],
  "top_features": [
    { "feature": "glucose", "value": 165, "shap": 0.42, "abs_shap": 0.42 },
    { "feature": "hba1c_level", "value": 7.2, "shap": 0.28, "abs_shap": 0.28 }
  ],
  "explain_note": "Las variables mostradas corresponden a los valores SHAP que explican la probabilidad base del modelo; la probabilidad final puede incluir reglas clínicas."
}
```

**Comportamientos importantes (no triviales):**

- **Aliasing glucosa**: `glucose` y `blood_glucose_level` se espejan automáticamente, así que enviar uno cubre al otro.
- **Cortocircuito de obesidad**: si `disease=obesidad` y el payload incluye `bmi`, el modelo ML se ignora y se aplica la regla OMS (BMI ≥ 30 → prob 0.99, sino 0.01). El response trae `method: "clinical_rule_bmi"` y `top_features: []`.
- **Reglas clínicas progresivas**: tras la inferencia, se aplica `prob = max(prob, regla)` por banda clínica (glucosa, HbA1c, presión arterial). Nunca bajan la probabilidad. El campo `raw_model_probability` preserva la probabilidad pre-reglas; `top_features` (SHAP) explica esa probabilidad cruda, no la final.
- **Filtro de género en SHAP**: las features `gender_*` se omiten del top-5 explicativo.
- **Features faltantes**: cualquier feature ausente se rellena con 0; los nombres no-dummy aparecen en `missing_filled_as_zero`.

Cada predicción se persiste en la tabla `predictions` (disease, input_data JSON, prediction, probability, timestamp).

### `GET /synthetic/<disease>`
Devuelve una fila aleatoria de los datos sintéticos curados (`data_curated/<disease>/<disease>_synthetic_ctgan*.csv`). Se usa en el frontend para autocompletar el simulador con un "caso clínico aleatorio". Si no hay sintéticos disponibles, hace fallback a `data_processed/`. El response incluye `_source_type: "synthetic"` o `"real"`.

---

## Estructura

```
chronic-risk-backend/
├── app.py                       # API Flask + carga de modelos + SHAP + reglas clínicas
├── prepare_datasets.py          # Normalización de CSVs crudos → esquema común
├── curate_and_synthesize.py     # Split estratificado + CTGAN/TVAE + diccionario de datos
├── train_models.py              # Entrenamiento de Regresión Logística por enfermedad
├── requirements.txt             # Dependencias (UTF-16)
├── runtime.txt                  # python-3.12.8
├── medical_history.db           # SQLite generada en runtime (gitignored)
├── data_raw/                    # CSVs públicos originales (PIMA, Kaggle, UCI, etc.)
├── data_processed/              # Dataset maestro por enfermedad
├── data_curated/                # Train/test split + sintéticos + diccionario por enfermedad
└── models/                      # *_pipeline.pkl + *_features.json + *_metrics.json
```

---

## Notas técnicas

- El pipeline sklearn tiene pasos nombrados **`scaler`** y **`clf`**. Estos nombres son load-bearing: SHAP los referencia explícitamente en `_build_shap_explainer`. Renombrarlos rompe la explicabilidad silenciosamente.
- Las 4 enfermedades son un set cerrado declarado en `FILES` (app.py) y en `DATASETS` (train_models.py). Para añadir una nueva, hay que tocar ambos archivos y entrenar el modelo correspondiente.
- `train_models.py` elimina features con leakage por target (quita `hypertension` para el modelo de hipertensión, `bmi` para obesidad, `heart_disease` para cardiovascular).
- `prepare_datasets.py` hace fan-in de 7 datasets públicos con esquemas distintos a un `COMMON_SCHEMA` único, con imputación por mediana y caps fisiológicos.
