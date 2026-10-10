# Predisana Frontend

SPA en React 19 + Vite que sirve como interfaz educativa de **Predisana**: una
plataforma para explorar la estimación de riesgo de enfermedades crónicas
(**diabetes, hipertensión y cardiovascular**) usando Machine Learning explicable.

Consume el API [`chronic-risk-backend`](../chronic-risk-backend) y presenta los
resultados con visualizaciones interactivas (SHAP, calibración, what-if, laboratorio
de datos sintéticos) sobre un sistema de diseño propio ("Pulso Sereno") con tema
claro/oscuro.

> ⚠️ Este sistema es puramente educativo y **NO** sustituye un diagnóstico médico.

> 📸 **Captura de pantalla de:** el simulador en modo oscuro con una predicción
> calculada (banda de riesgo + barras SHAP).

---

## Stack

- **React 19** + **react-router-dom 7** (SPA con rutas)
- **Vite 7** (dev server y build)
- **Bootstrap 5** + **react-bootstrap** (UI) + **react-bootstrap-icons** (iconos SVG)
- **recharts** (gráficos: SHAP, calibración, what-if, distribuciones, admin)
- **sweetalert2** (modales / alertas)
- **axios** (cliente HTTP)
- **ESLint 9** (lint)

---

## Requisitos

- Node.js **20.19+** (lo exige Vite 7; el CI usa 22)
- npm

---

## Instalación

```powershell
npm install
```

---

## Scripts

```powershell
npm run dev       # Vite dev server con HMR (http://localhost:5173)
npm run build     # build de producción en dist/
npm run preview   # sirve el build localmente para probarlo
npm run lint      # ESLint sobre todo el proyecto
npm test          # Vitest (58 tests, ~2 s)
npm run test:watch  # los mismos, en modo watch
```

---


## Tests

**58 tests con Vitest + Testing Library** (`npm test`), sobre las cosas del front que
tienen lógica de verdad:

- **`ErrorBoundary`** — que un fallo de render muestre una salida en vez de dejar la
  SPA en blanco, que registre el error en consola y que **«Reintentar» no deje la
  pantalla pegada**. Detalle aprendido escribiéndolos: el hijo de prueba no puede
  "fallar solo la primera vez", porque React reintenta el render al capturar un error
  y el test se recuperaría solo sin probar nada; el fallo se controla con un flag.
- **`getSessionId`** — que el UUID anónimo se persista y se reutilice (si se generase
  uno por petición, la analítica del admin contaría una sesión por click), que
  sobreviva al saneo del backend (`[A-Za-z0-9_-]`, 64 chars, AUD-8) y que haya
  fallback sin `crypto.randomUUID`. Más que `predictRisk` mande la cabecera
  `X-Session-Id` y el modo (el simplificado si no se pide otro), y que
  `evaluateSyntheticCase` evalúe al paciente del laboratorio con el modelo completo y
  lo marque con `?source=synthetic`, para que no cuente como una simulación.
- **Contrato de etiquetas** — lee los `*_features.json` del **backend** (los seis
  modelos de la v2) y la cabecera de los sintéticos (las columnas de las fichas del
  laboratorio, que traen más que los modelos, como el IMC autodeclarado) y exige que
  nada llegue a la UI sin etiqueta en español. Es el fallo silencioso que apareció al
  migrar hipertensión a NHANES: cambia el esquema y algo se pinta como
  `waist_circumference` en la ficha o en la barra de SHAP. Solo los one-hot heredan la
  etiqueta de su grupo (`gender_Male` → "Sexo"): con un prefijo cualquiera,
  `bmi_autodeclarado` pasaba el test por la etiqueta de `bmi`. Exige también que cada
  enfermedad servida tenga nombre y aparezca en las páginas que declaran su lista a mano
  (simulador, métricas, proyecto y educación). Y donde se pinta una
  columna one-hot suelta (métricas, laboratorio, admin), `getFeatureLabel` la lee como
  grupo y opción («Tabaquismo: Exfumador») en vez de `smoking_history_former`.
- **`AvisoSoporte`** — los avisos de cuánto fiarse del resultado, con la forma exacta
  que devuelve `/predict`. No todos traen `trained_range`: el nivel `incoherente`
  (peso, IMC y cintura que no cuadran entre sí, o una presión invertida) solo trae `detail`, y la primera
  versión, que lo desestructuraba siempre, tumbaba la página del simulador entera.
  Por encima del tope de edad de NHANES (80 = "80 o más") no dice que el modelo
  "nunca vio casos así": los vio, registrados como 80.
- **`riskBand`** — que el front pinte la banda bajo / moderado / alto que decide el
  backend (`risk_band` en `/predict`) y no la recalcule por tercios: los cortes van
  por enfermedad, porque en la v1 de diabetes (13,6% de media) un 31,8% salía como
  "Riesgo bajo". Sin ese campo cae a los tercios de siempre. Más la frase que explica de
  dónde salen los cortes (`bandNote`).
- **Paso de cada campo** (`fieldHints`) — que el paso de cada input admita los
  decimales del dato real, leyendo las fichas del sintético del backend. «Caso virtual»
  carga una en el formulario, y con la v2 la presión (media de tres lecturas: 101,4),
  la talla (172,7), el eGFR y la albúmina llegaban con más decimales que el paso: el
  navegador marcaba el campo como inválido y bloqueaba el envío sin decir nada.

Configuración en `vite.config.js` (los tests reusan los mismos alias y plugins que el
build) y arranque común en `src/test/setup.js`.

## Configuración

La URL del backend se controla con la variable de entorno **`VITE_API_URL`**. Si no
se define, hace fallback a `http://localhost:8000`.

Crea un archivo `.env` (o `.env.local`) en la raíz:

```env
VITE_API_URL=http://localhost:8000
```

Para producción (ej. Vercel), define `VITE_API_URL` en las variables de entorno del
entorno de despliegue apuntando al backend público.

El archivo `vercel.json` ya está configurado para servir la SPA con rewrites a
`index.html` (necesario para que react-router funcione en rutas profundas).

---

## Rutas / Páginas

| Ruta             | Componente       | Descripción |
|------------------|------------------|-------------|
| `/`              | `Home`           | Hero, propuesta de valor y metodología en 3 pasos. |
| `/educacion`     | `Educacion`      | Enciclopedia breve de las 3 enfermedades crónicas. |
| `/simulacion`    | `Simulacion`     | La página principal: formulario del modo simplificado o completo → riesgo + SHAP + capa clínica + what-if. |
| `/metricas`      | `Metricas`       | Por enfermedad y modo: leaderboard con el filtro de validación, AUC (también ponderado), calibración, bandas y reporte por clase. |
| `/proyecto`      | `Proyecto`       | Case study: historia de los datos, panorámica de modelos y laboratorio sintético (4 demos). |
| `/aviso`         | `Aviso`          | Aviso legal / disclaimer médico. |
| `/admin`         | `Admin`          | Dashboard de uso **dev-only**: sin link en la navbar, se accede por URL directa + token (`ADMIN_TOKEN` del backend). |

---

## Capa de servicios (`src/services/api.js`)

Cliente axios con base URL = `VITE_API_URL`. Expone:

**Núcleo del simulador**
- `checkHealth()` → `GET /health`
Cada enfermedad tiene dos modelos (v2) y las llamadas llevan `?mode=simplificado|completo`
(`MODOS`, `MODO_POR_DEFECTO = 'simplificado'`):

- `getConfig(disease, mode)` → `GET /config/<disease>` (campos del formulario, derivados, opcionales, límites aceptados, `not_used`, `topcoded`, bandas)
- `getMetrics(disease, mode)` → `GET /metrics/<disease>`
- `predictRisk(disease, payload, mode)` → `POST /predict/<disease>` — envía el header
  `X-Session-Id` con un **UUID anónimo** persistido en `localStorage`
  (`getSessionId()`), que agrupa simulaciones sin identificar a nadie.
- `getWhatIf(disease, {base, feature, min, max, steps, mode})` → `POST /whatif/<disease>`
  (curva contrafactual; el backend NO la registra en BD). En el simplificado, peso e IMC
  se barren a talla fija: la respuesta trae `coupled` y el panel lo explica bajo la curva.

**Laboratorio sintético (página Proyecto)**
- `getSyntheticCase(disease)` → `GET /synthetic/<disease>`
- `evaluateSyntheticCase(disease, payload, mode = 'completo')` → `POST /predict/<disease>?source=synthetic`
  (la ficha entera por el modelo completo, el que lee todas sus variables; el backend
  NO la registra en la BD)
- `getSampleCase(disease, source)` → `GET /sample/<disease>?source=real|synthetic`
- `getDistribution(disease, feature, bins)` → `GET /distribution/<disease>`
- `getSyntheticQuality(disease)` → `GET /synthetic_quality/<disease>`

**Panel admin** (todas mandan el header `X-Admin-Token`)
- `verifyAdmin(token)`, `getAdminStats(token, {from, to})`,
  `getAdminPredictions(token, {limit, disease, from, to})`,
  `downloadAdminCsv(token, opts)` (export CSV server-side)

> El historial de simulaciones es **server-side y anónimo**: vive en la tabla
> `predictions` del backend y se consulta desde el panel admin. No hay historial
> en `localStorage` (solo el UUID de sesión y el token admin en `sessionStorage`).

---

## Estructura

```
chronic-risk-frontend/
├── index.html
├── vite.config.js
├── vercel.json                  # rewrites SPA para Vercel
├── eslint.config.js
├── public/                      # assets estáticos
└── src/
    ├── main.jsx                 # entrypoint React
    ├── App.jsx                  # router + layout (navbar + footer)
    ├── index.css                # sistema de diseño "Pulso Sereno" (tokens, tema claro/oscuro)
    ├── components/
    │   ├── AvisoSoporte.jsx     # avisos de cobertura de datos / coherencia del resultado
    │   ├── ErrorBoundary.jsx    # captura fallos de render (dentro del Router, bajo la navbar)
    │   ├── Logo.jsx
    │   └── MyNavbar.jsx         # navegación + toggle de tema
    ├── context/
    │   └── ThemeContext.jsx     # tema claro/oscuro persistido
    ├── pages/                   # una página por ruta (7)
    ├── services/
    │   └── api.js               # cliente axios + sesión anónima + llamadas admin
    ├── test/
    │   └── setup.js             # arranque común de Vitest
    └── utils/
        ├── fieldHints.js        # ayuda y paso de cada campo del simulador
        ├── riskBand.js          # banda bajo/moderado/alto que decide el backend
        └── translations.js      # etiquetas en español (LABELS_ES, getLabel)
```

---

## Notas

- **App en español.** Toda la UI, etiquetas e identificadores siguen español
  (`hipertension`, `cardiovascular`). Mantener consistencia al editar.
- **Code-splitting por ruta (AUD-20).** Todas las páginas se cargan con
  `React.lazy()` salvo `Home`, que va en el bundle inicial a propósito por ser la
  primera pintura. Así recharts, sweetalert2 y el panel admin no se descargan hasta
  que se entra a la ruta que los usa. `vite.config.js` fija dos chunks a mano:
  `recharts` entero (partido entre las 4 páginas que lo usan, Rollup avisaba de un
  ciclo entre chunks) y `react` aparte, porque un chunk manual se queda con las
  dependencias que nadie reclama y sin eso recharts se llevaba React, con lo que la
  entrada descargaba los ~400 kB de gráficos en la primera pintura.
- **`ErrorBoundary`** se monta **dentro** del Router y **debajo** de la navbar, para
  que un fallo de una página no se lleve por delante la navegación. Su
  `key={pathname}` lo resetea al navegar; remontarlo no vuelve a pedir los chunks
  (`React.lazy` cachea el módulo ya resuelto), así que no deshace el code-splitting.
- **Las 5 enfermedades** (`diabetes`, `hipertension`, `cardiovascular`, `renal`, `higado`) son un
  set cerrado declarado en `DISEASES` dentro de `Simulacion.jsx` (y `Metricas.jsx` /
  `Proyecto.jsx`, más su ficha en `Educacion.jsx`). Si se añade una nueva al backend, hay
  que añadirla aquí también: el contrato de etiquetas lo comprueba contra los modelos
  servidos, así que una enfermedad que falte en una página tumba un test.
- **`Simulacion.jsx`** es la página más compleja: consume `/config`, `/predict`,
  `/synthetic` y `/whatif`, usa como min/max de los inputs los límites que sirve
  `/config.ranges` (los mismos con los que valida el API; `FIELD_HINTS`, en
  `src/utils/fieldHints.js`, solo guarda la ayuda y el paso de cada campo),
  renderiza el top-5 SHAP como barras de recharts y pinta la capa clínica
  (`clinical_flags`) aparte. Arriba se elige el **modo**: el simplificado pide lo que
  cualquiera sabe de sí mismo (con el peso y la talla, el backend calcula el IMC); el
  completo, además, las mediciones y analíticas del personal sanitario. Lo que define
  la enfermedad (HbA1c y glucosa en diabetes, presión en hipertensión) y lo que el
  modelo no usa (`not_used`) va en un bloque aparte marcado como opcional: no cambia
  la estimación, lo interpreta la guía.
- La respuesta de `/predict` distingue `probability` (calibrada con isotónica)
  de `raw_model_probability` (salida cruda del modelo, que es la que explica
  SHAP). La calibración es un reescalado monótono: no cambia el ranking.
- Los colores de los gráficos recharts van en **hex**, no `var()` CSS (recharts
  pinta atributos SVG donde las variables CSS no resuelven). Los tooltips sí se
  tematizan globalmente vía `.recharts-default-tooltip` en `index.css`.

---

## Despliegue

Pensado para Vercel (hay `vercel.json` listo). Cualquier hosting estático funciona:

```powershell
npm run build
# subir el contenido de dist/
```

Recordar definir `VITE_API_URL` apuntando al backend público antes del build.
