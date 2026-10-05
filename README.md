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

- **Simulador de riesgo** por enfermedad, en **dos modos**: el *simplificado* pide
  solo lo que cualquiera sabe de sí mismo (edad, sexo, tabaco, peso y talla,
  diagnósticos previos); el *completo*, pensado como apoyo para personal sanitario,
  suma mediciones y analíticas (cintura, presión, colesterol, HbA1c, función renal).
  Lo que define la enfermedad (la HbA1c en diabetes, la presión en hipertensión) se
  pide, pero lo interpreta la guía clínica, no el modelo.
- **Explicación de cada resultado**: top-5 de variables por impacto SHAP, más una
  capa de interpretación clínica (umbrales ADA / ACC-AHA) presentada aparte, sin
  alterar la salida del modelo.
- **Análisis "¿qué pasaría si...?"**: curva contrafactual de riesgo al variar una
  sola variable (p. ej. cómo cambia el riesgo con el peso, a la misma talla).

> 📸 **Captura de pantalla de:** el panel what-if — curva de riesgo al variar el
> peso, con el marcador en el valor actual del usuario.

- **Métricas en vivo y honestas**, por enfermedad y modo: leaderboard del bake-off
  (AUC de validación cruzada anidada, y qué candidatos no pasaron el filtro de
  validación y por qué), AUC de test también ponderado a la población, reporte por
  clase, **diagrama de fiabilidad** de la calibración y cómo reparten las **bandas
  del simulador** a la gente real del test.

> 📸 **Captura de pantalla de:** la página de métricas — leaderboard de algoritmos
> y curva de calibración, con el selector "Simplificado / Completo".

- **Laboratorio de datos sintéticos** (case study en `/proyecto`): generación de
  pacientes ficticios con CTGAN, juego "¿real o sintético?", distribuciones
  comparadas y **tres preguntas distintas** sobre el sintético — *fidelidad*
  (SDMetrics + heatmaps de correlación), *utilidad* (TSTR: entrenar solo con
  sintético y evaluar contra el test real, ratio 0,967-0,988) y *privacidad*
  (distancia al registro real más cercano, 1,07-1,22x la del propio test real).

> 📸 **Captura de pantalla de:** el laboratorio sintético — pestaña de
> distribuciones real vs sintético (o el juego "¿real o sintético?").

- **Panel admin dev-only** con analítica de uso agregada y anónima (KPIs,
  timeline, histogramas de probabilidad, features SHAP más frecuentes, export
  CSV), protegido por token. Los usuarios nunca se identifican: privacidad por
  diseño.

> 📸 **Captura de pantalla de:** el dashboard admin — KPIs, timeline diario y uso
> por enfermedad.

## Decisiones técnicas destacables

- **Datos reales, revisados antes de entrenar**: las tres enfermedades salen de
  **NHANES 2017-2023** (la encuesta de salud de los CDC, dos ciclos, 14 000-17 000
  adultos por enfermedad). La v1 mezclaba NHANES con un CSV de Kaggle para
  cardiovascular, donde los fumadores enfermaban menos y el modelo aprendía que fumar
  protege; antes aún se había descartado un dataset de hipertensión cuyo target era
  una fórmula del autor del CSV (el modelo daba 100% de riesgo a los 25 años).
- **Enfermedad total, no solo diagnosticada**: el 22-24% de quienes tienen diabetes y
  el 15-17% de quienes tienen hipertensión no estaban diagnosticados. El objetivo los
  cuenta (HbA1c, glucosa en ayunas, presión medida), así que el modelo estima tener la
  enfermedad, no que te la hayan diagnosticado.
- **Lo que define la enfermedad no entra al modelo**: con la HbA1c dentro, un modelo
  de diabetes solo reaprende el umbral diagnóstico. Esos datos se piden en el modo
  completo y los lee la capa clínica (ADA / ACC-AHA), que se devuelve como
  `clinical_flags` junto al número del modelo, nunca encima de él.
- **Seis modelos con filtro de validación**: por enfermedad y modo compiten LogReg,
  LightGBM y RandomForest con validación cruzada **anidada**, y solo puede ganar quien
  respeta el sentido clínico de cada variable para todas las personas del train (más
  edad, IMC o presión no pueden bajar el riesgo; más HDL no puede subirlo) y ordena
  bien cada subgrupo de sexo y edad. RandomForest no puede llevar restricciones y no
  pasa en ningún modo. AUC de test: 0,80-0,84 según enfermedad y modo.
- **Cuando los datos no dan señal, se dice**: en datos de un solo momento quien ya
  está tratado tiene la presión o el colesterol controlados, y quien enferma deja de
  fumar. Las restricciones impiden aprender la relación al revés, el efecto queda en
  cero, y si el usuario aporta esa variable la capa clínica le avisa de que esta
  estimación no la refleja.
- **Probabilidades calibradas**: isotónica centrada out-of-fold por modelo (Brier de
  cardiovascular simplificado 0,185 → 0,094), sin escalones extremos: ningún resultado
  vale 0% ni 100%, porque ningún grupo de personas del entrenamiento permite afirmar
  certeza. Tampoco hay mesetas: la curva une con rectas el centro de cada escalón, así
  que el what-if no sube a saltos. La API devuelve la probabilidad calibrada y la
  cruda, y SHAP explica la cruda — todo etiquetado.
- **Bandas de riesgo por enfermedad**: "bajo" es quedar por debajo de la media de
  los datos y "alto", al menos el doble, con los tercios (33% / 66%) como tope. Con
  tercios fijos, en la v1 una glucosa de 250 salía como "Riesgo bajo" en diabetes.
- **Más peso no baja el riesgo**: en la v1, con el mismo IMC y la misma cintura, más
  peso era más estatura, y la LogReg de hipertensión aprendía que eso protege. La v2
  no le da el peso a ningún modelo (el simplificado calcula el IMC con el peso y la
  talla) y el what-if barre el peso a talla fija.
- **Aviso de cobertura y de coherencia**: un modelo da un número igual de firme para
  una edad que vio miles de veces que para una que no vio nunca. `/predict` marca esas
  entradas, y las combinaciones que no pueden ser de una persona (una cintura que no
  cuadra con el IMC, una presión invertida), en `support_warnings` —sin tocar la
  probabilidad— y el what-if sombrea en la curva el tramo sin respaldo.
- **Capa de datos agnóstica al motor** (SQLAlchemy): SQLite en dev, Postgres en
  producción cambiando solo `DATABASE_URL`.
- **503 tests de pytest** sobre los invariantes delicados: signos clínicos de los
  modelos servidos, que `/metricas` publique lo que la API hace, calibración, capa
  clínica, avisos, laboratorio sintético y auth del admin.

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

  pipeline offline:  NHANES 2017-2023 → dataset limpio por enfermedad → reparto
  + bake-off anidado con filtro de validación → 6 modelos + calibradores → sintético CTGAN
```

## Páginas

| Ruta | Qué hay |
|---|---|
| `/` | Portada: propuesta de valor y metodología en 3 pasos |
| `/simulacion` | El simulador: formulario → riesgo + SHAP + capa clínica + what-if |
| `/metricas` | Por enfermedad y modo: leaderboard con el filtro de validación, calibración, bandas y reporte por clase |
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
npm test                 # 45 tests de Vitest (~2 s)
```

El frontend lee `VITE_API_URL` (por defecto `http://localhost:8000`). El panel
admin requiere definir la variable de entorno `ADMIN_TOKEN` en el backend.

**Tests del backend:**
```powershell
pip install -r requirements-dev.txt
python -m pytest         # 503 tests (BD temporal, no toca la de dev)
```

**CI:** cada push y pull request a `main` corre en GitHub Actions la suite de pytest
y el `lint` + `test` + `build` + `npm audit` del frontend (`.github/workflows/ci.yml`).

Consulta los README de cada subcarpeta para el detalle del pipeline de datos,
endpoints y convenciones.

## Licencia

Software propietario. © 2026 John Orellana. Todos los derechos reservados.
