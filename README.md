# Chronic Risk — Predicción educativa de riesgo de enfermedades crónicas

Aplicación full-stack que estima el riesgo de cuatro enfermedades crónicas
(**diabetes, hipertensión, obesidad y cardiovascular**) a partir de datos
clínicos, con un enfoque **educativo y explicable**: cada predicción se
acompaña de los factores que más influyeron (SHAP) y de reglas clínicas
transparentes.

> ⚠️ Proyecto con fines educativos y de portafolio. **No** es una herramienta
> de diagnóstico médico.

## Arquitectura

Monorepo con dos componentes:

| Carpeta | Stack | Rol |
|---|---|---|
| [`chronic-risk-backend/`](chronic-risk-backend/) | Python 3.12 · Flask · scikit-learn · SHAP · SDV | API REST + pipeline de datos y entrenamiento de modelos |
| [`chronic-risk-frontend/`](chronic-risk-frontend/) | React 19 · Vite · Bootstrap 5 · Recharts | SPA educativa que consume la API |

```
┌─────────────────┐      HTTP/JSON      ┌──────────────────────┐
│  React + Vite   │ ──────────────────▶ │  Flask REST API      │
│  (frontend SPA) │ ◀────────────────── │  modelos + SHAP      │
└─────────────────┘   probabilidad +    └──────────────────────┘
                      factores (SHAP)             │
                                                   ▼
                                      pipeline: datos → curado →
                                      sintético (CTGAN) → modelos
```

## Características

- **4 modelos** entrenados (LogisticRegression balanceada) servidos vía API.
- **Explicabilidad** con SHAP: top-5 factores por predicción.
- **Reglas clínicas** que complementan al modelo (umbrales de glucosa/HbA1c,
  bandas de presión sistólica, regla OMS de IMC ≥ 30).
- **Datos sintéticos** generados con CTGAN/TVAE (SDV) para demos sin exponer
  datos reales de pacientes.
- UI en español, orientada a explicar *por qué* de cada resultado.

## Puesta en marcha rápida

**Backend** (`chronic-risk-backend/`):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py            # http://localhost:8000
```

**Frontend** (`chronic-risk-frontend/`):
```powershell
npm install
npm run dev              # Vite dev server
```

El frontend lee `VITE_API_URL` (por defecto `http://localhost:8000`).

Consulta los README de cada subcarpeta para el detalle del pipeline de datos,
endpoints y convenciones.

## Licencia

Pendiente de definir.
