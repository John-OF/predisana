# Predisana Frontend

SPA en React 19 + Vite que sirve como interfaz educativa de **Predisana**: una plataforma para explorar la predicción de riesgo de enfermedades crónicas (diabetes, hipertensión, obesidad y riesgo cardiovascular) usando Machine Learning.

Consume el API [`chronic-risk-backend`](../chronic-risk-backend) y presenta los resultados con visualizaciones interactivas, explicabilidad SHAP y un historial de simulaciones local.

> ⚠️ Este sistema es puramente educativo y **NO** sustituye un diagnóstico médico.

---

## Stack

- **React 19** + **react-router-dom 7** (SPA con rutas)
- **Vite 7** (dev server y build)
- **Bootstrap 5** + **react-bootstrap** (UI)
- **recharts** (gráficos SHAP y métricas)
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

La URL del backend se controla con la variable de entorno **`VITE_API_URL`**. Si no se define, hace fallback a `http://localhost:8000`.

Crea un archivo `.env` (o `.env.local`) en la raíz:

```env
VITE_API_URL=http://localhost:8000
```

Para producción (ej. Vercel), define `VITE_API_URL` en las variables de entorno del entorno de despliegue apuntando al backend público.

El archivo `vercel.json` ya está configurado para servir la SPA con rewrites a `index.html` (necesario para que react-router funcione en rutas profundas).

---

## Rutas / Páginas

| Ruta             | Componente       | Descripción |
|------------------|------------------|-------------|
| `/`              | `Home`           | Hero, introducción al proyecto y accesos rápidos. |
| `/educacion`     | `Educacion`      | Contenido educativo sobre las 4 enfermedades crónicas. |
| `/simulacion`    | `Simulacion`     | Formulario clínico → predicción + SHAP. Es la página principal. |
| `/metricas`      | `Metricas`       | Métricas de los modelos (AUC, classification report train/test). |
| `/historial`     | `Historial`      | Historial de simulaciones del usuario (localStorage), con export JSON/CSV. |
| `/aviso`         | `Aviso`          | Aviso legal / disclaimer médico. |

---

## Capa de servicios

### `src/services/api.js`
Cliente axios con base URL = `VITE_API_URL`. Expone:

- `checkHealth()` → `GET /health`
- `getConfig(disease)` → `GET /config/<disease>`
- `getMetrics(disease)` → `GET /metrics/<disease>`
- `predictRisk(disease, payload)` → `POST /predict/<disease>`
- `getSyntheticCase(disease)` → `GET /synthetic/<disease>` (autocompleta el formulario con un caso clínico aleatorio)

### `src/services/historyService.js`
**El historial vive solo en `localStorage`** bajo la clave `chronic_sim_history_v1`. Es independiente de la tabla SQLite que mantiene el backend (esa no se expone por API). Funciones:

- `getHistory()`, `saveHistoryEntry(entry)`, `deleteHistoryEntry(id)`, `clearHistory()`
- `exportHistoryJSON()`, `exportHistoryCSV()` — descargas con timestamp.

---

## Estructura

```
chronic-risk-frontend/
├── index.html
├── vite.config.js
├── vercel.json                  # rewrites SPA para Vercel
├── eslint.config.js
├── public/                      # assets estáticos (íconos, imágenes)
└── src/
    ├── main.jsx                 # entrypoint React
    ├── App.jsx                  # router + layout (navbar + footer)
    ├── index.css                # estilos globales
    ├── App.css
    ├── components/
    │   └── MyNavbar.jsx         # navegación principal
    ├── pages/                   # una página por ruta
    ├── services/
    │   ├── api.js               # cliente axios
    │   └── historyService.js    # historial en localStorage
    ├── utils/
    │   └── translations.js      # etiquetas en español (LABELS_ES, getLabel)
    └── assets/
```

---

## Notas

- **App en español.** Toda la UI, etiquetas e identificadores siguen español (`hipertension`, `obesidad`). Mantener consistencia al editar.
- **Las 4 enfermedades** (`diabetes`, `hipertension`, `obesidad`, `cardiovascular`) son un set cerrado declarado en `DISEASES` dentro de `Simulacion.jsx`. Si se añade una nueva al backend, hay que añadirla aquí también.
- **`Simulacion.jsx`** es la página más compleja: consume `/config`, `/predict` y `/synthetic`, aplica los rangos `CLINICAL_LIMITS` a los inputs del formulario y renderiza el top-5 de SHAP como barras de recharts.
- La respuesta de `/predict` distingue `raw_model_probability` (probabilidad cruda del modelo, explicada por SHAP) de `probability` (probabilidad final tras reglas clínicas). El simulador muestra ambas para que el estudiante entienda la diferencia.

---

## Despliegue

Pensado para Vercel (hay `vercel.json` listo). Cualquier hosting estático funciona:

```powershell
npm run build
# subir el contenido de dist/
```

Recordar definir `VITE_API_URL` apuntando al backend público antes del build.
