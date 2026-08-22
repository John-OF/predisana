# Predisana — riesgo de enfermedades crónicas, explicado

Aplicación web full-stack que **estima el riesgo** de tres enfermedades crónicas
(**diabetes, hipertensión y cardiovascular**) con Machine Learning y — lo más
importante — **explica cada predicción**: qué variables la empujaron (SHAP), qué
dicen los umbrales clínicos de referencia (ADA / ACC-AHA) y qué tan honestas son
las probabilidades (calibración isotónica con curva de fiabilidad).

> ⚠️ Proyecto de portafolio con fines educativos y demostrativos. **No** es un
> dispositivo médico ni una herramienta de diagnóstico: es un estimador de
> riesgo tipo cribado, en la línea de los scores clínicos de autoevaluación.

<!-- TODO: añadir el link a la demo pública cuando esté desplegada (pendiente #7) -->

> 📸 **Captura de pantalla de:** el simulador con una predicción calculada —
> banda de riesgo, gráfico SHAP de los factores y nota clínica.

## Qué hace

- **Simulador de riesgo** por enfermedad, con formularios que solo piden datos
  que una persona común puede responder (autorreporte y medición casera). En
  diabetes la glucosa es **opcional**: sin ella se sirve un modelo de cribado; con
  ella, una variante más precisa (modelo híbrido).
- **Explicación de cada resultado**: top-5 de variables por impacto SHAP, más una
  capa de interpretación clínica (umbrales ADA / ACC-AHA) presentada aparte, sin
  alterar la salida del modelo.
- **Análisis "¿qué pasaría si...?"**: curva contrafactual de riesgo al variar una
  sola variable (p. ej. cómo cambia el riesgo a lo largo del rango de glucosa).

> 📸 **Captura de pantalla de:** el panel what-if — curva de riesgo al variar la
> glucosa, con el marcador en el valor actual del usuario.

- **Métricas en vivo y honestas**: leaderboard del bake-off de algoritmos por
  enfermedad (AUC de validación cruzada), reporte por clase y **diagrama de
  fiabilidad** de la calibración; en diabetes, comparación con/sin glucosa.

> 📸 **Captura de pantalla de:** la página de métricas — leaderboard de algoritmos
> y curva de calibración, con el toggle "Sin glucosa / Con glucosa".

- **Laboratorio de datos sintéticos** (case study en `/proyecto`): generación de
  pacientes ficticios con CTGAN, juego "¿real o sintético?", distribuciones
  comparadas y score de calidad SDMetrics con heatmaps de correlación.

> 📸 **Captura de pantalla de:** el laboratorio sintético — pestaña de
> distribuciones real vs sintético (o el juego "¿real o sintético?").

- **Panel admin dev-only** con analítica de uso agregada y anónima (KPIs,
  timeline, histogramas de probabilidad, features SHAP más frecuentes, export
  CSV), protegido por token. Los usuarios nunca se identifican: privacidad por
  diseño.

> 📸 **Captura de pantalla de:** el dashboard admin — KPIs, timeline diario y uso
> por enfermedad.

## Decisiones técnicas destacables

- **Datos reales por enfermedad** (sin frame maestro imputado): diabetes e
  hipertensión se entrenan con **NHANES 2021-2023** (encuesta real de los CDC);
  cardiovascular con el dataset público de Kaggle.
- **Auditoría de la señal, no solo del AUC**: el dataset previo de hipertensión se
  descartó al comprobar que su target era una fórmula del autor del CSV y no un
  desenlace clínico — el modelo la reaprendía y devolvía relaciones invertidas
  (100% de riesgo a los 25 años). Migrado a NHANES, el AUC baja de 0.95 a **0.80**
  y las relaciones son las clínicas: el riesgo crece con la edad y el IMC.
- **Modelo híbrido de diabetes**: variante self-report (LogReg, AUC 0.81, en el
  rango de los scores de cribado tipo FINDRISC) y variante con glucosa (LightGBM
  con restricción de monotonía, AUC 0.90). El backend rutea según lo que el
  usuario aporte.
- **Selección de modelos por CV**: por enfermedad compiten LogReg / RandomForest /
  LightGBM; se sirve el ganador y se publica el leaderboard completo.
- **Probabilidades calibradas**: isotónica out-of-fold por modelo (Brier de
  diabetes 0.111 → 0.042). La API devuelve la probabilidad calibrada y la cruda,
  y SHAP explica la cruda — todo etiquetado.
- **Capa clínica desacoplada**: los umbrales diagnósticos (ADA / ACC-AHA) se
  devuelven como `clinical_flags` junto al número del modelo, nunca encima de él.
- **Capa de datos agnóstica al motor** (SQLAlchemy): SQLite en dev, Postgres en
  producción cambiando solo `DATABASE_URL`.
- **148 tests de pytest** sobre los invariantes delicados: ruteo híbrido, alias de
  features, monotonía riesgo↔glucosa, calibración, capa clínica y auth del admin.

## Arquitectura

Monorepo con dos componentes:

| Carpeta | Stack | Rol |
|---|---|---|
| [`chronic-risk-backend/`](chronic-risk-backend/) | Python 3.12 · Flask · scikit-learn · LightGBM · SHAP · SDV · SQLAlchemy | API REST + pipeline de datos y entrenamiento |
| [`chronic-risk-frontend/`](chronic-risk-frontend/) | React 19 · Vite · Bootstrap 5 · Recharts | SPA educativa (tema claro/oscuro "Pulso Sereno") |

```
┌──────────────────┐       HTTP/JSON        ┌───────────────────────────┐
│  React + Vite    │ ─────────────────────▶ │  Flask REST API           │
│  (SPA educativa) │ ◀───────────────────── │  modelos + SHAP + capa    │
└──────────────────┘  prob. calibrada +     │  clínica + calibración    │
        │             SHAP + flags clínicos └───────────┬───────────────┘
        │  X-Admin-Token                                │
        ▼                                               ▼
┌──────────────────┐                        ┌───────────────────────────┐
│  Panel admin     │                        │  SQLAlchemy (predictions) │
│  (dev-only)      │ ◀───── analítica ───── │  SQLite dev / Postgres    │
└──────────────────┘        agregada        └───────────────────────────┘

  pipeline offline:  fuentes públicas (NHANES/Kaggle) → dataset limpio
  por enfermedad → split + sintético CTGAN → bake-off por CV → modelos + calibradores
```

## Páginas

| Ruta | Qué hay |
|---|---|
| `/` | Portada: propuesta de valor y metodología en 3 pasos |
| `/simulacion` | El simulador: formulario → riesgo + SHAP + capa clínica + what-if |
| `/metricas` | Leaderboard por enfermedad, calibración y reporte por clase |
| `/proyecto` | Case study: historia de los datos, modelos, laboratorio sintético, stack |
| `/educacion` | Enciclopedia breve de las tres enfermedades |
| `/aviso` | Aviso legal / disclaimer |
| `/admin` | Dashboard de uso (dev-only, por URL directa + token; sin link en la UI) |

## Puesta en marcha rápida

**Backend** (`chronic-risk-backend/`):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # runtime del API (el pipeline de datos va en
                                  # requirements-pipeline.txt: SDV/CTGAN, pesado)
python app.py                     # http://localhost:8000
```

**Frontend** (`chronic-risk-frontend/`):
```powershell
npm install
npm run dev              # Vite dev server (http://localhost:5173)
npm test                 # 20 tests de Vitest (~2 s)
```

El frontend lee `VITE_API_URL` (por defecto `http://localhost:8000`). El panel
admin requiere definir la variable de entorno `ADMIN_TOKEN` en el backend.

**Tests del backend:**
```powershell
pip install -r requirements-dev.txt
python -m pytest         # 148 tests, ~3 s (BD temporal, no toca la de dev)
```

**CI:** cada push y pull request a `main` corre en GitHub Actions la suite de pytest
y el `lint` + `test` + `build` + `npm audit` del frontend (`.github/workflows/ci.yml`).

Consulta los README de cada subcarpeta para el detalle del pipeline de datos,
endpoints y convenciones.

## Licencia

Software propietario. © 2026 John Orellana. Todos los derechos reservados.
