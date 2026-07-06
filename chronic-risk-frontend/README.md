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

- Node.js 18+ (recomendado 20+)
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
```

---

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
| `/simulacion`    | `Simulacion`     | La página principal: formulario clínico → riesgo + SHAP + capa clínica + what-if. |
| `/metricas`      | `Metricas`       | Leaderboard de algoritmos, curva de calibración y reporte por clase. En diabetes, toggle con/sin glucosa. |
| `/proyecto`      | `Proyecto`       | Case study: historia de los datos, panorámica de modelos y laboratorio sintético (4 demos). |
| `/aviso`         | `Aviso`          | Aviso legal / disclaimer médico. |
| `/admin`         | `Admin`          | Dashboard de uso **dev-only**: sin link en la navbar, se accede por URL directa + token (`ADMIN_TOKEN` del backend). |

---

## Capa de servicios (`src/services/api.js`)

Cliente axios con base URL = `VITE_API_URL`. Expone:

**Núcleo del simulador**
- `checkHealth()` → `GET /health`
- `getConfig(disease)` → `GET /config/<disease>` (features + opcionales + rangos)
- `getMetrics(disease)` → `GET /metrics/<disease>` (acepta también la variante `diabetes_glucosa`)
- `predictRisk(disease, payload)` → `POST /predict/<disease>` — envía el header
  `X-Session-Id` con un **UUID anónimo** persistido en `localStorage`
  (`getSessionId()`), que agrupa simulaciones sin identificar a nadie.
- `getWhatIf(disease, {base, feature, min, max, steps})` → `POST /whatif/<disease>`
  (curva contrafactual; el backend NO la registra en BD)

**Laboratorio sintético (página Proyecto)**
- `getSyntheticCase(disease)` → `GET /synthetic/<disease>`
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
    │   ├── Logo.jsx
    │   └── MyNavbar.jsx         # navegación + toggle de tema
    ├── context/
    │   └── ThemeContext.jsx     # tema claro/oscuro persistido
    ├── pages/                   # una página por ruta (7)
    ├── services/
    │   └── api.js               # cliente axios + sesión anónima + llamadas admin
    └── utils/
        └── translations.js      # etiquetas en español (LABELS_ES, getLabel)
```

---

## Notas

- **App en español.** Toda la UI, etiquetas e identificadores siguen español
  (`hipertension`, `cardiovascular`). Mantener consistencia al editar.
- **Las 3 enfermedades** (`diabetes`, `hipertension`, `cardiovascular`) son un set
  cerrado declarado en `DISEASES` dentro de `Simulacion.jsx` (y `Metricas.jsx` /
  `Proyecto.jsx`). Si se añade una nueva al backend, hay que añadirla aquí también.
- **`Simulacion.jsx`** es la página más compleja: consume `/config`, `/predict`,
  `/synthetic` y `/whatif`, aplica los rangos `CLINICAL_LIMITS` a los inputs,
  renderiza el top-5 SHAP como barras de recharts y pinta la capa clínica
  (`clinical_flags`) aparte. En diabetes, la **glucosa es un campo opcional**: si
  el usuario la aporta, el backend sirve la variante híbrida más precisa y la UI
  lo indica con un badge.
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
