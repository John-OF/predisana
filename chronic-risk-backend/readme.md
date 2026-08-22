# Predisana Backend

API REST en Python/Flask que sirve modelos de Machine Learning para la estimación de riesgo de enfermedades crónicas (diabetes, hipertensión y riesgo cardiovascular). Es el backend de una plataforma educativa: integra inferencia con **calibración isotónica**, explicabilidad con **SHAP**, un **modelo híbrido de diabetes** (con/sin glucosa), análisis contrafactual (**what-if**), generación de datos sintéticos con **CTGAN/TVAE**, una capa de interpretación clínica (**ADA / ACC-AHA**), y registro anónimo de uso sobre **SQLAlchemy** (SQLite en dev, Postgres en prod) consultable desde un panel admin protegido por token.

---

## Características

- **3 modelos servidos desde `models/`** (uno por enfermedad). Cada modelo es el **ganador de un bake-off por validación cruzada** (LogReg / RandomForest / LightGBM); no todas las enfermedades usan el mismo algoritmo.
- **Modelo híbrido de diabetes (NHANES 2021-2023)** — dos variantes entrenadas sobre datos reales de los CDC: `diabetes` (solo autorreporte, LogReg, AUC 0.81) y `diabetes_glucosa` (+glucosa sérica, LightGBM con restricción de monotonía, AUC 0.90). `/predict` rutea automáticamente a la variante con glucosa si el usuario la aporta.
- **Calibración isotónica por modelo** — la probabilidad servida es honesta (Brier de diabetes 0.111 → 0.042); la API devuelve la calibrada y la cruda, y la curva de fiabilidad queda en `_metrics.json`.
- **Selección de features por respondibilidad** — el simulador es educativo / de autoevaluación, así que cada modelo usa solo features que una persona común puede responder (autorreporte y medición casera/farmacia); la glucosa de diabetes es la única semi-accesible y es **opcional**.
- **Explicabilidad SHAP agnóstica al modelo** — `LinearExplainer` para modelos lineales y `TreeExplainer` para árboles. Cada predicción devuelve el top-5 de variables con mayor impacto.
- **Capa de interpretación clínica desacoplada** — la probabilidad reportada es la salida del modelo (calibrada); los umbrales diagnósticos (ADA para glucosa/HbA1c, ACC/AHA para presión sistólica) se devuelven aparte como `clinical_flags`/`clinical_note` y **no** modifican la probabilidad.
- **Análisis contrafactual (`/whatif`)** — barre una variable sobre un rango y devuelve la curva de riesgo, sin registrar nada en la BD.
- **Generación de datos sintéticos** con SDV (CTGAN por defecto, TVAE opcional) + endpoints de comparación real vs sintético (muestras, distribuciones, calidad SDMetrics) que alimentan el laboratorio del frontend.
- **Pipeline de datos por enfermedad** — desde fuentes públicas a un dataset limpio por enfermedad (sin frame maestro concatenado ni imputación cruzada).
- **Registro anónimo de uso** sobre SQLAlchemy (`DATABASE_URL`: SQLite en dev, Postgres en prod con el mismo código) + **panel admin dev-only** con analítica agregada, protegido por `X-Admin-Token`.
- **Suite de 108 tests (pytest)** sobre los invariantes delicados de la API.

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

# Instalar dependencias del API
pip install -r requirements.txt

# Solo si vas a regenerar datos/sintéticos/modelos (añade SDV+CTGAN y sdmetrics)
pip install -r requirements-pipeline.txt
```

> `requirements.txt` es el **runtime del API** y lista solo dependencias directas (las
> transitivas las resuelve pip). SDV/CTGAN viven aparte en `requirements-pipeline.txt`
> porque arrastran torch (~479 MB) y el servidor no los necesita: el informe de calidad
> del sintético se precomputa en el pipeline y la API sirve el JSON.

---

## Uso

### Levantar la API

```powershell
python app.py
# Servidor en http://0.0.0.0:8000

# Debugger interactivo de Werkzeug (opt-in, NUNCA en producción):
$env:FLASK_DEBUG = "1"; python app.py
```

Producción (con `gunicorn`, ya incluido en requirements):

```bash
gunicorn app:app
```

Al arrancar, `app.py` ejecuta automáticamente:
1. `_load_all()` — carga los pipelines de `models/` (las 3 enfermedades + la variante `diabetes_glucosa`), sus calibradores isotónicos, lee el modelo ganador de cada `_metrics.json` y construye un explainer SHAP acorde al tipo de cada modelo (Linear o Tree).
2. `init_db()` — crea la tabla `predictions` vía SQLAlchemy (por defecto `sqlite:///medical_history.db`; con `DATABASE_URL` apunta a Postgres u otro motor) y migra columnas nuevas si la BD venía del esquema viejo.

Variables de entorno: `DATABASE_URL` (motor de BD), `ADMIN_TOKEN` (habilita los endpoints
`/admin/*`; sin ella responden 503), `FLASK_DEBUG` (debugger local) y las de CORS / rate limiting
(`CORS_ORIGINS`, `ADMIN_CORS_ORIGINS`, `RATE_LIMIT_*`, `TRUST_PROXY_HEADERS`). Plantilla comentada
en `.env.example`.

**CORS y rate limiting.** Los endpoints públicos quedan abiertos si no se define `CORS_ORIGINS`
(cómodo en dev y para probar con curl); en producción se le pasa el dominio del front. `/admin/*`
**no hereda** ese `*`: solo acepta los orígenes de `ADMIN_CORS_ORIGINS` (por defecto, localhost de
desarrollo) y además rechaza con **403** cualquier `Origin` fuera de la lista — CORS por sí solo
únicamente le oculta la respuesta al navegador, la petición se ejecuta igual. Los límites son **por
IP**: `RATE_LIMIT_DEFAULT` global, `RATE_LIMIT_PREDICT` en `/predict` y `/whatif`, `RATE_LIMIT_ADMIN`
en el panel y uno estricto (`RATE_LIMIT_ADMIN_VERIFY`, 10/min) en `/admin/verify`, que es contra lo
que se fuerza-brutea el token. `/health` está exento para no romper los monitores de uptime. Detrás
de un proxy hay que activar `TRUST_PROXY_HEADERS=1` (si no, todo el tráfico comparte una sola
cubeta); sin proxy delante, activarlo permitiría falsear la IP con `X-Forwarded-For`.

### Pipeline de datos y entrenamiento

Si no tienes los modelos entrenados (o quieres regenerarlos), corre los scripts en
orden. Los datos crudos **no están versionados**: `data_raw/README.md` documenta
cada fuente y de dónde descargarla.

```powershell
# 1. Datasets limpios por enfermedad (data_raw/ -> data_processed/)
python prepare_datasets.py              # cardiovascular (Kaggle)
python prepare_nhanes_diabetes.py       # diabetes (NHANES 2021-2023)
python prepare_nhanes_hipertension.py   # hipertensión (NHANES 2021-2023)

# 2. Split estratificado + síntesis CTGAN/TVAE por enfermedad
python curate_and_synthesize.py

# 2b. Informe de calidad del sintético (SDMetrics) precomputado a JSON
python build_quality_reports.py

# Opciones útiles
python curate_and_synthesize.py `
    --model ctgan `
    --synth_multiplier 1.0 `
    --balance `
    --steps 15000 `
    --only diabetes,hipertension

# 3. Entrenamiento
python train_models.py                # bake-off por CV: hipertensión y cardiovascular
python train_models.py --only cardiovascular   # solo una (no re-serializa la otra)
python train_nhanes_diabetes.py       # diabetes híbrida: variantes con/sin glucosa + calibradores
```

Diabetes e hipertensión se generan con sus propios scripts NHANES:
`prepare_datasets.py` ya **no las toca** (solo construye cardiovascular), así que
correrlo completo no pisa esos datasets. `curate_and_synthesize.py` y
`train_models.py` sí operan sobre ellos, porque leen el CSV ya generado.

**Hipertensión (AUD-1):** el dataset venía de un CSV de ENSANUT cuyo target
`riesgo_hipertension` era una fórmula del autor, no un desenlace clínico. El modelo
la reaprendía y devolvía relaciones invertidas (a más edad, menos riesgo; 100% de
riesgo para un joven sano). Migrado a NHANES: prevalencia realista (36%), gradiente
correcto por edad (4,8% a los 18-27 → 62% a los 78-87) e IMC, y AUC honesto de
**0.80** en vez del 0.95 que producía la fórmula memorizada. La presión arterial
**no es feature** del modelo —sería un umbral disfrazado— sino un dato opcional que
alimenta la capa clínica ACC/AHA.

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
Liveness + comprobación real de la BD (`SELECT 1`). Devuelve **503** si la base no responde.

```json
{
  "status": "ok",
  "database": "sqlite",
  "database_ok": true,
  "models_loaded": ["cardiovascular", "diabetes", "diabetes_glucosa", "hipertension"]
}
```

### `GET /metrics/<disease>`
Devuelve `models/<disease>_metrics.json`. Incluye el modelo ganador (`best_model`), el `leaderboard` con el AUC de CV de cada candidato, el classification report de test y la curva de calibración (`calibration`: Brier crudo/calibrado + puntos de la curva de fiabilidad). Acepta también la variante **`diabetes_glucosa`** (el modelo híbrido con glucosa).

### `GET /config/<disease>`
Configuración para construir el formulario en el frontend:

```json
{
  "disease": "diabetes",
  "features": ["age", "bmi", "hypertension", "..."],
  "optional_features": ["blood_glucose_level"],
  "ranges": { "age": [18, 100], "bmi": [15, 50], "glucose": [60, 260], "blood_pressure": [60, 130] },
  "categoricals": { "gender": ["Female", "Male"], "smoking_history": ["current", "former", "never"] }
}
```

Las `features` son las propias del esquema de cada enfermedad (heterogéneo), ya recortadas por respondibilidad. Las `optional_features` son las que aporta una variante con más datos (hoy: la glucosa del híbrido de diabetes) y el usuario puede rellenar o no. Las `categoricals` se derivan de los nombres de las columnas one-hot (`gender_*`, `smoking_history_*`).

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
  "variant": "glucosa",
  "used_glucose": true,
  "probability": 0.72,
  "raw_model_probability": 0.81,
  "calibrated": true,
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
  "explain_note": "La probabilidad mostrada es la salida del modelo calibrada (isotónica); `raw_model_probability` es la salida cruda. Los valores SHAP explican el modelo crudo. Los indicadores clínicos (ADA/ACC-AHA) se muestran aparte como referencia y NO modifican la probabilidad."
}
```

**Comportamientos importantes (no triviales):**

- **Ruteo híbrido de diabetes**: si el payload trae una glucosa válida (>0) y existe la variante `diabetes_glucosa`, se sirve esa (`variant: "glucosa"`, `used_glucose: true`); si no, el modelo self-report (`variant: "base"`).
- **Aliasing glucosa**: `glucose` y `blood_glucose_level` se espejan automáticamente, así que enviar uno cubre al otro.
- **Calibración**: `probability` es la salida del modelo **calibrada** con la isotónica persistida; `raw_model_probability` es la cruda (la que SHAP explica). La isotónica es un reescalado monótono: no cambia el ranking (AUC intacto).
- **Capa de interpretación clínica desacoplada (A4)**: `clinical_flags`/`clinical_note` exponen los umbrales diagnósticos de referencia (ADA: glucosa ≥100/≥126/≥200, HbA1c ≥5.7/≥6.5; ACC/AHA: sistólica ≥120/≥130/≥140/≥180). Esta capa **no** altera la probabilidad (sustituye al antiguo `max()` con números mágicos).
- **Filtro de género en SHAP**: las features `gender_*` se omiten del top-5 explicativo.
- **Features faltantes**: cualquier feature ausente se rellena con 0; los nombres no-dummy aparecen en `missing_filled_as_zero`.

Cada predicción se persiste (anónima) en la tabla `predictions`: disease, input_data JSON, prediction, probability, modelo servido, nota clínica, top SHAP y el `session_id` que el frontend manda en el header `X-Session-Id` (UUID aleatorio: agrupa sin identificar).

### `POST /whatif/<disease>`
Análisis contrafactual: recibe `{ base, feature, min, max, steps }`, fija el caso `base` y barre `feature` sobre el rango, devolviendo la curva `[{value, probability, raw_probability}]` (calibrada y cruda). Usa el mismo ruteo híbrido que `/predict`. **No** registra nada en la BD ni calcula SHAP.

### `GET /synthetic/<disease>`
Devuelve una fila aleatoria de los datos sintéticos curados (`data_curated/<disease>/<disease>_synthetic_ctgan*.csv`). Se usa en el frontend para autocompletar el simulador con un "caso clínico aleatorio". Si no hay sintéticos disponibles, hace fallback a `data_processed/`. El response incluye `_source_type: "synthetic"` o `"real"`.

### `GET /sample/<disease>?source=real|synthetic`
Una ficha de paciente del origen pedido, en formato homogéneo. Alimenta el juego "¿real o sintético?" del laboratorio.

### `GET /distribution/<disease>?feature=<col>&bins=<n>`
Histograma comparado real vs sintético de una variable numérica, sobre bins comunes y normalizado a % (compara la *forma* aunque difiera el tamaño de muestra).

### `GET /synthetic_quality/<disease>`
Score de fidelidad del sintético (SDMetrics `QualityReport`: overall, column shapes, pair trends, detalle por columna) + matrices de correlación real/sintético para el heatmap comparado. El informe se **precomputa** en el pipeline (`build_quality_reports.py` → `data_curated/<enfermedad>/<enfermedad>_quality.json`) y la API lo sirve tal cual, así que producción no necesita `sdmetrics` (que arrastra torch). Solo lo recalcula si el JSON falta y la librería está instalada.

### Admin (dev-only): `GET /admin/verify` · `/admin/stats` · `/admin/predictions` · `/admin/export.csv`
Protegidos por el header `X-Admin-Token`, que debe coincidir con la env var `ADMIN_TOKEN` (sin ella responden **503**; token incorrecto, **401**). No es auth de usuario — los usuarios nunca se loguean. Además: `Origin` no permitido → **403**, y más de `RATE_LIMIT_ADMIN_VERIFY` intentos de token por minuto y por IP → **429**.

- `/admin/stats` — analítica **agregada y anónima**: totales, sesiones únicas, conteo/tasa de positivos/probabilidad media por enfermedad, histograma de probabilidades, timeline diario, uso por hora y features SHAP más frecuentes. Acepta `?from=YYYY-MM-DD&to=YYYY-MM-DD`.
- `/admin/predictions?limit=&disease=&from=&to=` — simulaciones recientes.
- `/admin/export.csv` — export CSV server-side con los mismos filtros.

---

## Estructura

```
chronic-risk-backend/
├── app.py                       # API Flask: modelos + SHAP + calibración + capa clínica + admin
├── prepare_datasets.py          # CSVs crudos → dataset limpio (hipertensión, cardiovascular)
├── prepare_nhanes_diabetes.py   # NHANES 2021-2023 (.xpt) → dataset de diabetes
├── prepare_nhanes_hipertension.py # NHANES 2021-2023 (.xpt) → dataset de hipertensión
├── curate_and_synthesize.py     # Split estratificado + síntesis CTGAN/TVAE
├── build_quality_reports.py     # Precomputa el informe de calidad del sintético a JSON
├── synthetic_quality.py         # Cálculo SDMetrics + correlaciones (pipeline y fallback del API)
├── train_models.py              # Bake-off multi-modelo por CV (hipertensión, cardiovascular)
├── train_nhanes_diabetes.py     # Diabetes híbrida: variantes con/sin glucosa + calibradores
├── tests/                       # Suite pytest (108 tests; BD temporal propia)
├── pytest.ini
├── .env.example                 # Plantilla de variables de entorno
├── requirements.txt             # Runtime del API (directas, UTF-8)
├── requirements-pipeline.txt    # + SDV/CTGAN y sdmetrics (solo pipeline de datos)
├── requirements-dev.txt         # + pytest
├── runtime.txt                  # python-3.12.8
├── medical_history.db           # SQLite generada en runtime (gitignored)
├── data_raw/                    # Fuentes crudas (NO versionadas; ver data_raw/README.md)
├── data_processed/              # Un dataset limpio por enfermedad
├── data_curated/                # Train/test split + sintéticos por enfermedad
└── models/                      # *_pipeline.pkl + *_calibrator.pkl + *_features.json + *_metrics.json
```

---

## Notas técnicas

- El pipeline sklearn tiene pasos nombrados **`scaler`** y **`clf`**. Estos nombres son load-bearing: SHAP los referencia explícitamente en `_build_shap_explainer`. Renombrarlos rompe la explicabilidad silenciosamente.
- Las 3 enfermedades son un set cerrado declarado en `FILES` (app.py) y en `DATASETS` (train_models.py). Para añadir una nueva, hay que tocar ambos archivos y entrenar el modelo correspondiente. Las **variantes de modelo extra** (hoy `diabetes_glucosa`) se declaran en `EXTRA_MODEL_KEYS`/`VARIANT_BASE` en app.py: se cargan y se sirven en `/metrics`, pero no son enfermedades por sí solas (no entran en la UI ni en el admin como categoría propia).
- `train_models.py` corre un bake-off por enfermedad (LogReg / RandomForest / LightGBM con restricciones de monotonía clínica), elige el mejor por AUC en cross-validation, ajusta la calibración isotónica out-of-fold y persiste ganador + calibrador; el score de cada candidato queda en `_metrics.json.leaderboard`.
- `get_features_for_disease()` parte del esquema propio de cada dataset y solo descarta las features de laboratorio (no respondibles, `DROP_NON_RESPONDABLE`). Con los esquemas por-enfermedad (B1) ya no hay columnas de leakage que recortar.
- `prepare_datasets.py` produce un dataset limpio por enfermedad desde su fuente cruda, con imputación por mediana *intra-dataset* y caps fisiológicos (sin esquema común ni imputación cruzada). Para diabetes, la fuente canónica es NHANES vía `prepare_nhanes_diabetes.py`.
- **El entrenamiento del GAN se mide en pasos, no en épocas (AUD-24).** CTGAN hace `ceil(n_filas / 500)` actualizaciones por época, así que un `--epochs` fijo daba **10x menos entrenamiento** a un dataset de 4,8k filas que a uno de 55k. Se notaba: a 50 épocas la correlación peso↔cintura del sintético de hipertensión era **−0,08** frente a **+0,90** en el real, y el **22%** de las filas eran físicamente imposibles (peso <75 kg con cintura >115 cm). El parámetro real es **`--steps`** (default 15000) y las épocas se derivan por dataset; `--epochs` sigue estando para forzarlas a mano. Entrenar más tampoco es gratis: el *training-by-sampling* de CTGAN **aplana las marginales categóricas** según avanza (cardiovascular a 15k pasos se iba del 35% de hombres real al 48,5%), así que `PASOS_POR_DATASET` deja esa enfermedad en 5500, que es donde ya estaba bien. Y acerca al GAN a memorizar filas reales, así que se comprobó: la distancia mediana de un sintético a su vecino real más cercano se queda en **1,3-1,8x** la del propio test real, y las colisiones exactas (3 de 54392 en cardiovascular, donde el dato es grueso) son **150x más raras** que entre el train y el test reales.
- **Los one-hot se colapsan antes de entrenar el GAN (AUD-13).** CTGAN veía `gender_Male` y `gender_Female` como dos binarias independientes y las sampleaba por separado: salían pacientes sintéticos **sin género** o **con los dos a la vez** (solo el 46% de las filas de diabetes eran válidas), algo que se ve a simple vista en la ficha del laboratorio y hace trivial el juego "¿real o sintético?". `curate_and_synthesize.py` detecta los grupos mutuamente excluyentes sobre los datos reales, los convierte en **una sola columna categórica** para el ajuste y los expande de vuelta al muestrear (round-trip exacto, con test). No se arregla con un argmax a posteriori: eso *inventa* una categoría donde el GAN no eligió ninguna. Efecto medido en el informe SDMetrics: diabetes **0.754 → 0.823**, hipertensión **0.733 → 0.850**, cardiovascular **0.892 → 0.924** (la ganancia está sobre todo en *column pair trends*, que es justo lo que rompía la correlación imposible entre las dummies).
- El engine de BD se construye desde **`DATABASE_URL`** (default `sqlite:///medical_history.db`); para Postgres en producción basta cambiar la env var, mismo código.
