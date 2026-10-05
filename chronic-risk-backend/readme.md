# Predisana Backend

API REST en Python/Flask que sirve modelos de Machine Learning para estimar el riesgo de tres enfermedades crónicas (diabetes, hipertensión y enfermedad cardiovascular). Es el backend de una plataforma educativa: integra inferencia con **calibración isotónica centrada**, explicabilidad con **SHAP**, **dos modos por enfermedad** (simplificado y completo), análisis contrafactual (**what-if**), generación de datos sintéticos con **CTGAN**, una capa de interpretación clínica (**ADA / ACC-AHA**) y registro anónimo de uso sobre **SQLAlchemy** (SQLite en dev, Postgres en prod) consultable desde un panel admin protegido por token.

---

## Características

- **Datos reales de NHANES 2017-2023** (CDC, EE. UU.) para las tres enfermedades: dos ciclos juntos, 14 000-17 000 adultos por enfermedad, con su peso muestral.
- **Enfermedad total, no solo diagnosticada**: el objetivo cuenta a quien lo tiene aunque no lo sepa (diabetes: diagnosticada, HbA1c ≥ 6,5% o glucosa en ayunas ≥ 126 mg/dL; hipertensión: diagnosticada, ≥ 140/90 mmHg medida o medicación; cardiovascular: cardiopatía coronaria, angina, infarto, insuficiencia cardíaca o ictus autorreportados).
- **Dos modos por enfermedad, seis modelos**: el **simplificado** pide lo que cualquiera sabe de sí mismo (edad, sexo, tabaco, peso y talla, diagnósticos previos); el **completo**, además, lo que mide el personal sanitario (IMC y cintura medidos, presión, colesterol total y HDL, HbA1c, eGFR y albúmina en orina).
- **Lo que define la enfermedad no entra al modelo**: la HbA1c y la glucosa en diabetes, o la presión en hipertensión. Con ellas el modelo solo reaprendería el umbral diagnóstico. Se piden igual (en el completo) y las interpreta la capa clínica con las guías.
- **Cada modelo es el ganador de un bake-off con validación cruzada anidada** (LogReg / LightGBM / RandomForest) **y un filtro de validación**: ningún candidato puede invertir el sentido clínico de una variable ni ordenar mal un subgrupo (sexo, tramo de edad). Se publica el leaderboard completo, con los descartados y su motivo.
- **Probabilidades calibradas** (isotónica centrada out-of-fold), que nunca valen 0% ni 100% y no suben a saltos; la API devuelve la calibrada y la cruda.
- **Explicabilidad SHAP** (`LinearExplainer` para la LogReg, `TreeExplainer` para LightGBM): top-5 de variables por predicción.
- **Capa clínica, cobertura de datos y coherencia desacopladas**: umbrales ADA / ACC-AHA, avisos de extrapolación y de datos que no cuadran entre sí. Nada de eso toca la probabilidad.
- **Análisis contrafactual (`/whatif`)**, **laboratorio de datos sintéticos** (CTGAN + informe de fidelidad, utilidad y privacidad) y **registro anónimo de uso** con panel admin.
- **Suite de 503 tests (pytest)** sobre los invariantes delicados de la API, los modelos y los datos.

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
1. `_load_all()` — carga de `models/` los seis pipelines (`<enfermedad>_<modo>`), sus calibradores, el modelo ganador y las variables sin efecto de cada `_metrics.json`, y construye el explainer SHAP de cada uno. Del train curado saca la cobertura de datos de cada modo, las medianas de lo que el modelo no usa y la prevalencia de cada enfermedad. Si falta algún modelo lo dice en el log y `/health` responde 503.
2. `init_db()` — crea la tabla `predictions` vía SQLAlchemy (por defecto SQLite en `medical_history.db`; con `DATABASE_URL` apunta a Postgres u otro motor) y migra columnas nuevas si la BD venía de un esquema viejo.

Las rutas (`models/`, `data_curated/`, `data_processed/` y esa SQLite) cuelgan de la carpeta del
backend, no del directorio desde el que se arranque: antes, lanzada desde la raíz del repo, la app
levantaba sin modelos ni datos y `/health` decía "ok".

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
en cada endpoint del panel y uno estricto para los intentos de token **fallidos**
(`RATE_LIMIT_ADMIN_VERIFY`, 10/min), compartido por los cuatro endpoints del panel: todos comprueban el
token, así que cualquiera sirve para fuerza-brutearlo. `/health` está exento para no romper los
monitores de uptime. Detrás de un proxy hay que activar `TRUST_PROXY_HEADERS=1` (si no, todo el
tráfico comparte una sola cubeta); sin proxy delante, activarlo permitiría falsear la IP con
`X-Forwarded-For`.

### Pipeline de datos y entrenamiento

Si quieres regenerar los datos o los modelos, corre los scripts en orden. Los datos
crudos **no están versionados**: `data_raw/README.md` documenta de dónde descargarlos.

```powershell
# 1. Un dataset limpio por enfermedad (data_raw/nhanes/ -> data_processed/)
python prepare_nhanes.py

# 2. Reparto train/test por enfermedad + bake-off de los dos modos (-> data_curated/, models/)
python train_models.py                    # ~4 min
python train_models.py --only diabetes    # una sola enfermedad
python train_models.py --rapido           # rejillas y pliegues mínimos, para depurar

# 3. Sintético CTGAN del laboratorio, sobre ese train (20-40 min en CPU)
python curate_and_synthesize.py
python curate_and_synthesize.py --only hipertension --steps 15000 --oversample 4

# 4. Informe de calidad del sintético (SDMetrics + TSTR + privacidad) precomputado a JSON
python build_quality_reports.py
```

**Los datos (`prepare_nhanes.py`).** Dos ciclos de NHANES, 2017-marzo 2020 prepandemia y
2021-2023, con el peso muestral repartido según la duración de cada uno (3,2 y 2 años). Se
excluye a menores de 18 años y a embarazadas, y los "no sabe" / "se niega" son dato
faltante, no un "no". Solo entra quien tiene la medición que define su objetivo (HbA1c,
presión): si no, los no diagnosticados sin analítica contarían como sanos. Resultado:
diabetes 14 347 personas (18,9% con la enfermedad), hipertensión 14 017 (44,3%) y
cardiovascular 16 815 (12,8%). Decisiones con su porqué en la cabecera del script:

- **Enfermedad total**: según el ciclo, el 22-24% de quienes tienen diabetes y el 15-17%
  de quienes tienen hipertensión no estaban diagnosticados. Entrenar con "alguna vez se
  lo dijeron" enseñaba al modelo a predecir el diagnóstico, no la enfermedad.
- **La glucosa es la de ayunas** (`LBXGLU`). La v1 usaba la del perfil bioquímico, con y
  sin ayuno mezclados, y la app la leía con umbrales de ayunas.
- **El simplificado usa el peso y la talla autodeclarados** (`WHQ`), que es lo que el
  usuario va a escribir: la gente se quita IMC (−0,8 de media, −1,5 con obesidad y −2,3
  con un IMC de 40 o más). El
  completo usa el IMC medido.
- **Sin actividad física ni alcohol**: el cuestionario de actividad cambió entre ciclos
  (cumplir los 150 min/semana de la OMS daba 34% en un ciclo y 53% en el otro), y en
  datos de un solo momento quien enferma deja de beber y sale "protector".

**El entrenamiento (`train_models.py`).** Un solo reparto por enfermedad (80/20,
estratificado por objetivo y ciclo) que comparten los dos modos; cada modo se queda con
las filas que tienen todas sus variables. Compiten LogReg, LightGBM y RandomForest, con
los hiperparámetros ajustados por **validación cruzada anidada** (5 pliegues exteriores ×
3 interiores: la búsqueda se repite dentro de cada pliegue, así que el AUC con el que se
comparan no está inflado por haber elegido los parámetros sobre los mismos datos). Solo
puede ganar quien pasa el **filtro de validación**:

- **Signos clínicos**: subir una variable con sentido +1 (edad, IMC, cintura, presión,
  colesterol, HbA1c, albúmina, diagnósticos, fumar o haber fumado) no puede bajar el
  riesgo de nadie del train, ni subirlo una con sentido −1 (HDL, eGFR). LogReg
  (`MonotonicLogisticRegression`) y LightGBM (`monotone_constraints`) se entrenan con
  esas restricciones; RandomForest no puede llevarlas y no pasa el filtro en ningún modo.
- **Subgrupos**: AUC ≥ 0,55 por sexo y por tramo de edad (18-39, 40-59, 60-80), para que
  no gane un modelo que solo ordena por edad.

El ganador se calibra sobre sus predicciones out-of-fold y se evalúa **una vez** en el
test, también con los pesos muestrales de NHANES (representativo de los adultos de
EE. UU.):

| Enfermedad | Modo | Modelo | AUC test | AUC ponderado | Brier crudo → calibrado |
|---|---|---|---|---|---|
| Diabetes | simplificado | LightGBM | 0,801 | 0,826 | 0,189 → 0,126 |
| Diabetes | completo | LightGBM | 0,835 | 0,871 | 0,168 → 0,113 |
| Hipertensión | simplificado | LightGBM | 0,820 | 0,811 | 0,174 → 0,172 |
| Hipertensión | completo | LightGBM | 0,828 | 0,824 | 0,170 → 0,168 |
| Cardiovascular | simplificado | LightGBM | 0,823 | 0,840 | 0,185 → 0,094 |
| Cardiovascular | completo | LogReg | 0,829 | 0,848 | 0,183 → 0,088 |

**Variables sin efecto.** Las restricciones impiden que un modelo aprenda una relación al
revés, y si los datos no dan señal en el sentido clínico el efecto queda en cero exacto:
en datos de un solo momento, quien ya está diagnosticado suele estar tratado
(antihipertensivos, estatinas) o ha dejado de fumar. Pasa con el tabaco en diabetes
(haber fumado en el simplificado; fumar y haber fumado en el completo) y con la presión,
el colesterol total y la HbA1c en el completo de cardiovascular. Quedan en
`_metrics.json.sin_efecto`, `/config` las sirve como `not_used` y, si el usuario las
aporta, la capa clínica lo dice. Un test exige que de verdad pesen cero.

### Tests

La suite de `pytest` cubre los invariantes delicados: que cada modelo servido respete
los signos clínicos y lo que publica `/metricas` (sensibilidad, bandas, tamaño del test),
la calibración sin extremos ni mesetas, la capa clínica, los avisos de cobertura y de
coherencia, el what-if, los límites de entrada, el laboratorio sintético (one-hot,
coherencia, relaciones fuertes, decimales, marginales) y la auth y el rate limiting del
admin. También comprueba contra los modelos servidos lo que afirman los textos del
frontend: las variables que, según la página de Educación, más pesan en cada modelo
(`test_textos_educacion.py`). Corre contra una base SQLite temporal (nunca toca
`medical_history.db`).

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

---

## Endpoints

Todos los endpoints aceptan/devuelven JSON. Los de una enfermedad aceptan
`?mode=simplificado|completo` (sin él, el simplificado; otro valor es 400). El CORS de los
endpoints públicos depende de `CORS_ORIGINS` (sin definir = abierto, cómodo en dev);
`/admin/*` es un recurso aparte que **nunca** hereda ese `*` — ver la sección de CORS y
rate limiting más arriba.

### `GET /health`
Liveness + comprobación real de la BD (`SELECT 1`) y de los modelos. Devuelve **503** (`"status": "degraded"`) si la base no responde o si falta alguno de los seis modelos (`models_missing`): sin el modelo de un modo, el `/predict` de ese modo responde 500 aunque el otro funcione. Está exento del rate limiting (un 429 marcaría el deploy como caído ante un monitor de uptime), así que el resultado de la BD se reutiliza **5 s**: un bucle contra `/health` no se traduce en una consulta por petición.

```json
{
  "status": "ok",
  "database": "sqlite",
  "database_ok": true,
  "models_loaded": ["cardiovascular_completo", "cardiovascular_simplificado", "diabetes_completo",
                    "diabetes_simplificado", "hipertension_completo", "hipertension_simplificado"],
  "models_missing": []
}
```

### `GET /metrics/<disease>?mode=`
Devuelve `models/<disease>_<modo>_metrics.json`: el objetivo, `n_train`/`n_test` y la prevalencia, el modelo ganador (`best_model`), el `leaderboard` (AUC anidado, mejores parámetros, violaciones de signo, AUC por subgrupo y si pasa el filtro), las variables `sin_efecto`, el AUC de test (también ponderado) y por subgrupo, el classification report de test (la clasificación del propio modelo: salida cruda ≥ 0,5, la misma regla que el campo `prediction` de `/predict`), la calibración (`calibration`: Brier crudo/calibrado + curvas de fiabilidad) y `bands`: los cortes de las bandas del simulador y cómo reparte cada una a la gente real del test (`share`, `positive_rate`, `share_of_positives`).

### `GET /config/<disease>?mode=`
Configuración para construir el formulario en el frontend (diabetes, simplificado):

```json
{
  "disease": "diabetes",
  "mode": "simplificado",
  "modes": ["simplificado", "completo"],
  "features": ["age", "gender_Male", "gender_Female", "smoking_history_current", "smoking_history_former",
               "bmi", "hypertension", "high_cholesterol", "heart_disease"],
  "inputs": ["age", "gender", "smoking_history", "weight", "height", "hypertension", "high_cholesterol", "heart_disease"],
  "derived": { "bmi": ["weight", "height"] },
  "optional_features": ["blood_glucose_level", "hba1c_level", "ap_hi", "ap_lo"],
  "clinical_inputs": ["blood_glucose_level", "hba1c_level", "ap_hi", "ap_lo"],
  "defining_inputs": ["hba1c_level", "blood_glucose_level"],
  "not_used": ["smoking_history_former"],
  "ranges": { "age": [18, 100], "weight": [25, 300], "height": [100, 230], "bmi": [10, 95], "...": "..." },
  "feature_support": { "age": { "min": 18.0, "max": 80.0, "p1": 18.0, "p99": 80.0, "n": 11016 }, "...": "..." },
  "topcoded": { "age": 80 },
  "risk_bands": { "low_below": 0.1892, "high_from": 0.3785, "prevalence": 0.1892, "relative_to_prevalence": true },
  "categoricals": { "gender": ["Female", "Male"], "smoking_history": ["current", "former", "never"] }
}
```

`features` son las columnas del modelo, en su orden; `inputs`, los campos del formulario: los grupos one-hot van por su nombre (sus opciones, en `categoricals`) y lo que la API calcula va por sus entradas (`derived`: en el simplificado el IMC sale del peso y la talla). `clinical_inputs` son los datos que solo lee la capa clínica y **no** cambian la estimación; `defining_inputs`, los que definen la enfermedad (el completo los pide y los lee la guía). `not_used`, las variables que el modelo acepta pero no usa.

`ranges` y `feature_support` **no son lo mismo**. `ranges` son los límites **físicos** que acepta el API (`INPUT_LIMITS`): fuera de ellos `/predict` y `/whatif` responden 400, y el formulario los usa como min/max de sus inputs, así que front y back validan con los mismos números. `feature_support` es el tramo que cada variable continua realmente cubre en el train de ese modo: dentro de `ranges` pero fuera de ahí, el valor se acepta con un aviso (AUD-16), y la UI lo usa para sombrear el what-if donde el modelo extrapola. `topcoded` marca las variables con tope en los datos (NHANES registra a todo mayor de 80 como 80). `risk_bands` son los cortes de las bandas bajo / moderado / alto de la enfermedad (ver *Bandas de riesgo* en `/predict`).

### `POST /predict/<disease>?mode=`
Recibe las variables del modo y devuelve la probabilidad de riesgo + explicación SHAP + capa clínica.

**Request** (diabetes, simplificado; la HbA1c es opcional y solo la lee la guía):
```json
{
  "age": 55, "weight": 92, "height": 172,
  "hypertension": 1, "high_cholesterol": 0, "heart_disease": 0,
  "gender_Male": 1, "gender_Female": 0,
  "smoking_history_never": 0, "smoking_history_current": 0, "smoking_history_former": 1,
  "hba1c_level": 6.1
}
```

**Response:**
```json
{
  "disease": "diabetes",
  "mode": "simplificado",
  "model": "lightgbm",
  "probability": 0.2638,
  "raw_model_probability": 0.6087,
  "calibrated": true,
  "risk_band": "mid",
  "risk_bands": { "low_below": 0.1892, "high_from": 0.3785, "prevalence": 0.1892, "relative_to_prevalence": true },
  "prediction": 1,
  "bmi": 31.1,
  "top_features": [
    { "feature": "age", "value": 55.0, "shap": 0.623, "abs_shap": 0.623 },
    { "feature": "bmi", "value": 31.1, "shap": 0.397, "abs_shap": 0.397 }
  ],
  "clinical_flags": [
    { "indicator": "hba1c", "value": 6.1, "category": "prediabetes", "source": "ADA",
      "detail": "HbA1c 5.7–6.4%: rango de prediabetes." },
    { "indicator": "sin_efecto", "value": ["smoking_history_former"], "category": "no_reflejado_en_el_modelo",
      "source": "modelo", "detail": "Indicaste que fumas o has fumado. El tabaco es un factor de riesgo, pero esta estimación no lo refleja: ..." }
  ],
  "clinical_note": "HbA1c 5.7–6.4%: rango de prediabetes. Indicaste que fumas o has fumado. ...",
  "not_used": ["smoking_history_former"],
  "support_warnings": [],
  "support_note": "Todos los valores caen dentro del rango con datos de entrenamiento.",
  "explain_note": "La probabilidad mostrada es la salida del modelo calibrada (isotónica centrada); ..."
}
```

**Comportamientos importantes (no triviales):**

- **Todo lo que el modelo usa es obligatorio**: si falta, la respuesta es **400** y dice qué falta (`faltan datos del modo simplificado: age`; en el simplificado, el IMC se puede dar o sacar del peso y la talla). La v1 rellenaba con 0 lo ausente, y un IMC de 0 no es un paciente. Lo único que puede faltar son las variables sin efecto, que toman la mediana del train: no cambian nada. Cada grupo (sexo, tabaco) tiene que traer una categoría y solo una.
- **Límites físicos (`INPUT_LIMITS`)**: todo valor que se aporta, feature o dato clínico, tiene que caer dentro de su rango (edad 18-100, peso 25-300 kg, IMC 10-95, glucosa 30-700…) o la respuesta es **400** con el rango aceptado. Son topes de lo físicamente posible, no el rango entrenado: cubren todas las filas reales y sintéticas (hay un test que lo exige). Antes una edad de −30 con un IMC de 900 daba 200, probabilidad 1,0 y una fila en la BD. Un peso y una talla que dan un IMC fuera de su rango también son 400.
- **Calibración**: `probability` es la salida del modelo **calibrada**; `raw_model_probability` es la cruda (la que SHAP explica). Es un reescalado monótono: no cambia el ranking (AUC intacto). **Sin extremos**: la isotónica a secas termina siempre en 0,0 y en 1,0 — su primer escalón son los scores más bajos hasta el primer enfermo y el último, los más altos desde el último sano, tengan los puntos que tengan — y en la v1 de diabetes el 100% lo sostenía **una sola persona** del train. `fit_calibrator()` funde cada escalón extremo puro con su vecino y le da la tasa real de los dos juntos. **Sin mesetas**: la curva une con rectas el centro de cada escalón con su nivel (la isotónica *centrada*, Oron y Flournoy 2017); con escalones el what-if subía a saltos. Rangos servidos en la v2: diabetes 0,5%–83,3% (simplificado) y 0,2%–94,3% (completo), hipertensión 0,8%–92,5% y 1,2%–97,9%, cardiovascular 0,3%–62,1% y 0,4%–75,0%.
- **Bandas de riesgo por enfermedad**: `risk_band` (`low` / `mid` / `high`) es la lectura de `probability` frente a la media de la enfermedad, y `risk_bands` trae los cortes. **"Bajo" es quedar por debajo de la media de los datos de entrenamiento y "alto", al menos el doble**, con los tercios (33% / 66%) como tope. Los dos modos de una enfermedad comparten los cortes: diabetes 18,9% / 37,8%, cardiovascular 12,8% / 25,6%; hipertensión, con un 44,3% de prevalencia, conserva los tercios. Con tercios fijos, en la v1 una mujer sana de 25 años con glucosa de 250 salía **"Riesgo bajo"** con un 31,8%, más del doble de la media. Medido en el test real (diabetes, simplificado): bajo = 58,1% de la gente con un 6,8% de diabéticos, moderado = 27,4% con un 29,2%, alto = 14,4% con un 47,6%. Igual que la capa clínica, la banda **no** altera la probabilidad ni `prediction`. `/metricas` publica el reparto de cada modo con los mismos cortes que sirve la API (hasta la revisión de la v2, el entrenamiento los sacaba de las filas de cada modo: en cardiovascular completo publicaba 11,9% / 23,8% y servía 12,8% / 25,6%).
- **`prediction` es la clase del modelo**: 1 cuando la salida **cruda** llega a 0,5 (lo mismo que `pipeline.predict`), que es exactamente la regla que evalúa el entrenamiento y que `/metricas` publica como sensibilidad. Los modelos se entrenan con clases balanceadas y la calibración deshace ese balanceo, así que cortar la **calibrada** en 0,5 es otro clasificador: en diabetes (simplificado) daría una sensibilidad del 17,4% frente al 78,5% publicado. Consecuencia a tener presente: `prediction` puede valer 1 con una `probability` del 26%, como en el ejemplo de arriba. Son tres lecturas distintas y las tres van etiquetadas: `probability` (calibrada), `risk_band` (frente a la media de la enfermedad) y `prediction` (la clase del modelo). El simulador enseña las dos primeras; `prediction` solo se ve en el panel admin.
- **Capa de interpretación clínica desacoplada (A4)**: `clinical_flags`/`clinical_note` exponen los umbrales diagnósticos de referencia (ADA: glucosa en ayunas ≥ 100/≥ 126/≥ 200, HbA1c ≥ 5,7/≥ 6,5; ACC/AHA: la más alta de sistólica ≥ 120/≥ 130/≥ 140/≥ 180 y diastólica ≥ 80/≥ 90/≥ 120). Esta capa **no** altera la probabilidad. Los datos que solo la alimentan (glucosa, HbA1c y presión cuando el modelo no los usa) se validan igual que las features: no numérico, no finito o fuera de rango → 400. Además, si el usuario marca o aporta una variable **sin efecto**, un indicador `sin_efecto` le dice que esta estimación no la refleja y por qué.
- **Filtro de SHAP**: las features `gender_*` (no son un factor sobre el que actuar) y las variables sin efecto (su SHAP es 0) se omiten del top-5.
- **Aviso de cobertura de datos (AUD-16)**: `support_warnings`/`support_note` señalan las entradas que caen donde el modelo de ese modo tiene pocos datos (`pocos_datos`, fuera del p1-p99) o ninguno (`sin_datos`, fuera del min-max observado). Por encima del tope de edad de NHANES (80) el aviso lleva `topcoded` y no dice que el modelo "no vio ningún caso": sí los vio, pero registrados como 80. Las variables sin efecto no avisan: no mueven la estimación.
- **Coherencia entre campos**: el nivel `incoherente` de `support_warnings` marca combinaciones que no pueden ser de una misma persona aunque cada dato caiga en su rango (`coherence.py`): una cintura que no cuadra con el IMC, una diastólica igual o mayor que la sistólica, o, en el simplificado, un IMC dado a mano que con ese peso implica una talla imposible. En el completo el peso no se juzga: no es dato del modo, y el laboratorio manda la ficha entera, con el peso **declarado** junto al IMC **medido** — cruzar dos mediciones distintas daba una talla implícita de menos de 1,30 m a 9 de cada 9 500 personas reales. Ese nivel no trae `trained_range`, solo `detail`.

Cada predicción se persiste (anónima) en la tabla `predictions`: disease, modo, las variables que el modelo o la capa clínica usan (más el peso y la talla del simplificado; nunca claves arbitrarias del cliente), prediction, probability, modelo servido, nota clínica, top SHAP y el `session_id` que el frontend manda en el header `X-Session-Id` (UUID aleatorio: agrupa sin identificar). La excepción es `?source=synthetic`, con el que el laboratorio evalúa los pacientes del GAN: la respuesta es la misma, pero no se guarda, porque no es la simulación de nadie y mezclaría la analítica del admin con datos sintéticos.

### `POST /whatif/<disease>?mode=`
Análisis contrafactual: recibe `{ base, feature, min, max, steps }`, fija el caso `base` y barre `feature` sobre el rango, devolviendo la curva `[{value, probability, raw_probability}]` (calibrada y cruda), más el `supported_range` de la variable barrida (el tramo con datos de entrenamiento detrás, que el frontend sombrea: ahí la curva se aplana por falta de datos, no porque el riesgo deje de subir) y `topcoded_at` si la variable tiene tope en los datos. Usa los mismos límites que `/predict` (un barrido fuera de `ranges` es 400). Solo barre variables continuas que el modelo usa de verdad: cualquier otra (las que definen la enfermedad, las sin efecto, las de otro modo) es 400, porque daría una recta. **No** registra nada en la BD ni calcula SHAP.

Una excepción al "se barre una sola variable": en el simplificado, `weight` y `bmi` se barren **a talla fija**. Barrer el peso con el IMC quieto es barrer la **estatura**, y en la v1 la curva de hipertensión bajaba (59% a 45 kg, 27% a 140 kg). Ahora la talla sale del caso base y, al mover uno, el otro lo sigue; sin talla, barrer el peso es 400. La respuesta lo declara en `coupled: {feature, height_m}` y cada punto trae su `coupled_value` (topado a los límites físicos, para que un valor derivado no devuelva 400); `coupled` es `null` en cualquier otro barrido.

### `GET /synthetic/<disease>`
Devuelve una fila aleatoria de los datos sintéticos curados (`data_curated/<disease>/<disease>_synthetic_ctgan*.csv`), con las columnas del laboratorio (`modos.columnas_laboratorio`). Se usa en el frontend para el "caso virtual" del laboratorio. El response incluye `_source_type: "synthetic"` o `"real"`.

### `GET /sample/<disease>?source=real|synthetic`
Una ficha de paciente del origen pedido, en formato homogéneo. Alimenta el juego "¿real o sintético?" del laboratorio.

### `GET /distribution/<disease>?feature=<col>&bins=<n>`
Histograma comparado real vs sintético de una columna del laboratorio, sobre bins comunes y normalizado a % (compara la *forma* aunque difiera el tamaño de muestra).

`/synthetic`, `/sample` y `/distribution` leen los CSV **una sola vez** y los mantienen en memoria (`_read_csv_cached`). Los CSV no cambian con el servidor en marcha; el pipeline los regenera offline y un reinicio los recarga.

### `GET /synthetic_quality/<disease>`
Tres preguntas sobre el sintético, no una. **Fidelidad**: SDMetrics `QualityReport` (overall, column shapes, pair trends, detalle por columna) + matrices de correlación real/sintético para el heatmap. **Utilidad** (`tstr`): se entrena un modelo **solo con sintético** con las variables del modo completo y se evalúa contra el *test real*, junto al mismo modelo entrenado con datos reales sobre ese mismo test — los **mismos dos algoritmos en las dos ramas**, para que la diferencia sea de los datos y no del modelo. Ratios actuales: **0,967-0,988**. **Privacidad** (`privacy`): distancia al registro real más cercano (DCR). La referencia **no es cero** — el propio test real también está cerca del train, así que se reportan las dos; el sintético queda **1,07-1,22x** más lejos que el test real. Las copias exactas se cuentan en el sintético *y* entre los reales. El informe se **precomputa** en el pipeline (`build_quality_reports.py` → `data_curated/<enfermedad>/<enfermedad>_quality.json`) y la API lo sirve tal cual, así que producción no necesita `sdmetrics` (que arrastra torch). Solo lo recalcula si el JSON falta y la librería está instalada.

### Admin (dev-only): `GET /admin/verify` · `/admin/stats` · `/admin/predictions` · `/admin/export.csv`
Protegidos por el header `X-Admin-Token`, que debe coincidir con la env var `ADMIN_TOKEN` (sin ella responden **503**; token incorrecto, **401**). No es auth de usuario — los usuarios nunca se loguean. Además: `Origin` no permitido → **403**, y más de `RATE_LIMIT_ADMIN_VERIFY` intentos de token **fallidos** por minuto y por IP, sumando los cuatro endpoints → **429** en todo el panel, también con el token bueno (si no, el acierto se distinguiría de los fallos). El uso con el token bueno no gasta ese cupo; cada endpoint tiene además su `RATE_LIMIT_ADMIN`.

- `/admin/stats` — analítica **agregada y anónima**: totales, sesiones únicas, conteo/tasa de positivos/probabilidad media por enfermedad, histograma de probabilidades, timeline diario, uso por hora y features SHAP más frecuentes. Trae el modelo de cada modo. Acepta `?from=YYYY-MM-DD&to=YYYY-MM-DD`.
- `/admin/predictions?limit=&disease=&from=&to=` — simulaciones recientes, con su modo. `limit` va de 1 a 500 (50 por defecto, también si llega ≤ 0: SQLite lee `LIMIT -1` como "sin límite").
- `/admin/export.csv` — export CSV server-side con los mismos filtros.

---

## Estructura

```
chronic-risk-backend/
├── app.py                       # API Flask: modelos + SHAP + calibración + capa clínica + admin
├── modos.py                     # Enfermedades, modos y variables de cada modelo (fuente única)
├── prepare_nhanes.py            # NHANES 2017-2023 (.xpt) → un dataset limpio por enfermedad
├── train_models.py              # Reparto + bake-off anidado con filtro de validación + calibración
├── monotonic_logreg.py          # LogReg que respeta el signo clínico de cada variable (la carga el API)
├── risk_banding.py              # Regla de las bandas bajo/moderado/alto (API + entrenamiento)
├── coherence.py                 # Datos que no cuadran entre sí (aviso de /predict + filtro del sintético)
├── curate_and_synthesize.py     # Síntesis CTGAN/TVAE sobre el train de cada enfermedad
├── build_quality_reports.py     # Precomputa el informe de calidad del sintético a JSON
├── synthetic_quality.py         # SDMetrics + correlaciones + TSTR + privacidad
├── tests/                       # Suite pytest (503 tests; BD temporal propia)
├── pytest.ini
├── .env.example                 # Plantilla de variables de entorno
├── requirements.txt             # Runtime del API (directas, UTF-8)
├── requirements-pipeline.txt    # + SDV/CTGAN y sdmetrics (solo pipeline de datos)
├── requirements-dev.txt         # + pytest
├── runtime.txt                  # python-3.12.8
├── medical_history.db           # SQLite generada en runtime (gitignored)
├── data_raw/                    # Fuentes crudas (NO versionadas; ver data_raw/README.md)
├── data_processed/              # Un dataset limpio por enfermedad
├── data_curated/                # Train/test por enfermedad + sintético + informe de calidad
└── models/                      # <enfermedad>_<modo>_{pipeline.pkl, calibrator.pkl, features.json, metrics.json}
```

---

## Notas técnicas

- **`modos.py` es la fuente única** de qué es cada modelo: enfermedades, modos, variables de cada uno (`FEATURES`), las que definen la enfermedad (`DEFINITORIAS`), el sentido clínico de cada variable (`SENTIDO`), de qué columna sale cada una (`FUENTE`: el `bmi` del simplificado es el IMC autodeclarado), lo que la API calcula (`DERIVADAS`), lo que va en escala log (`LOG`: la albúmina) y las columnas de las fichas del laboratorio. La usan la ingesta, el entrenamiento, el sintético y la API. Para añadir una enfermedad: su objetivo en `prepare_nhanes.py`, su entrada en `modos.py`, entrenar, y declararla en el frontend (`DISEASES` de `Simulacion.jsx`, `Metricas.jsx` y `Proyecto.jsx`).
- El pipeline sklearn tiene pasos nombrados **`prep`** (log de la albúmina), **`scaler`** y **`clf`**. SHAP explica lo que ve el clasificador: la salida de `pipe[:-1]`. Renombrar los pasos rompe la explicabilidad.
- **`MonotonicLogisticRegression`** (`monotonic_logreg.py`): la feature que sale con el signo contrario al clínico se reajusta fuera y vuelve con coeficiente **cero exacto**, que con una pérdida convexa es el óptimo con esa cota. Nació en la v1, donde la LogReg de hipertensión aprendió, por colinealidad entre peso, IMC y cintura, que pesar más protegía (un hombre de 50 años con IMC 30 daba 26,7% con 70 kg y 13,0% con 100 kg); en la v2 ningún modelo recibe el peso y las restricciones van en todos.
- **El tabaco va contra "nunca"**: `smoking_history_current` y `smoking_history_former` con sentido +1 y "nunca" como referencia. Con las tres columnas, la de "nunca" no tiene sentido clínico que imponer y el modelo podía hacer que haber fumado restara riesgo (en diabetes, −2,3 puntos para un exfumador).
- **Un solo reparto por enfermedad (AUD-15).** En la v1, `train_nhanes_diabetes.py` hacía su propio `train_test_split` mientras `data_curated/` salía de otro script: el AUC publicado no se medía sobre el test que sirve el laboratorio y el **fondo de SHAP** contenía el **79%** de las filas de test de la variante con glucosa. Ahora `train_models.py` escribe el único reparto y todo lo demás lo lee; los tests exigen que train y test sean disjuntos y que el fondo de SHAP, la cobertura y las medianas salgan de las mismas filas con las que se entrenó cada modelo.
- **El GAN ve las columnas del laboratorio** (`modos.columnas_laboratorio`) de las filas completas del train: las del modo completo, las que definen la enfermedad (menos la glucosa, que solo está en la submuestra en ayunas) y el peso, la talla y el IMC autodeclarados.
- **El entrenamiento del GAN se mide en pasos, no en épocas (AUD-24).** CTGAN hace `ceil(n_filas / 500)` actualizaciones por época, así que un `--epochs` fijo daba **10x menos entrenamiento** a un dataset pequeño que a uno grande (en la v1, a 50 épocas la correlación peso↔cintura del sintético de hipertensión era −0,08 frente a +0,90 en el real). El parámetro real es **`--steps`** (default 15 000, donde satura) y las épocas se derivan por dataset. La semilla fija numpy y torch: la misma semilla da el mismo sintético, y el ruido que queda es el que hay entre semillas (no merece la pena afinar sobre diferencias < 0,05). Las marginales categóricas se desviaban entre 0,05 y 0,18 con cualquier número de pasos: se corrigen **después de muestrear**, generando un pool mayor (`--oversample`, 4 por defecto) y eligiendo filas para que la distribución **conjunta** de lo categórico (y el pico de edad 80) coincida con la real, con cupos por resto mayor. **Para contarlo bien:** así las marginales categóricas quedan *impuestas*, no aprendidas; el ajuste solo **elige** filas, nunca las inventa ni las repondera.
- **Los one-hot se colapsan antes de entrenar el GAN (AUD-13).** CTGAN veía `gender_Male` y `gender_Female` como dos binarias independientes: salían pacientes **sin sexo** o **con los dos a la vez** (en la v1 solo el 46% de las filas de diabetes eran válidas). `curate_and_synthesize.py` convierte cada grupo en **una sola columna categórica** para el ajuste y la expande al muestrear (round-trip exacto, con test).
- **Relaciones fuertes del sintético.** CTGAN reproduce bien cada columna y mal las relaciones fuertes entre dos, y entrenar más no lo arregla. En la v1 de hipertensión peso~IMC salía a 0,64 (real 0,89) y de ahí 100 pacientes con una talla implícita de 0,94 a 2,59 m. Ahora no se le pide al GAN esa relación: el peso se deriva como IMC autodeclarado × talla², y la cintura y el IMC autodeclarado se modelan como **residuo** de una recta sobre el IMC medido, igual que la diastólica sobre la sistólica (el GAN dejaba su correlación en 0,30 frente al 0,61 real; con el residuo sale 0,62 en diabetes, pero 0,43-0,45 en hipertensión y cardiovascular, donde el GAN aprende una relación entre el residuo y la sistólica que en el real es cero: queda abierto). Se recomponen al muestrear y se topan al rango real. **Para contarlo bien:** esas correlaciones vienen en buena parte por construcción, no aprendidas. Además, el generador descarta del pool las filas que el simulador marcaría como incoherentes, redondea cada columna a los decimales que usa el 99% del real (el peso salía con dos decimales y el real lleva uno) y trata "estar en el tope de edad" como un estrato más al elegir filas.
- El engine de BD se construye desde **`DATABASE_URL`** (default: SQLite en `medical_history.db`, dentro de la carpeta del backend); para Postgres en producción basta cambiar la env var, mismo código.
