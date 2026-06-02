# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a thesis project ("Tesis-MP") split into two sibling projects, each with its own git repo:

- `chronic-risk-backend/` — Flask REST API + ML training pipeline (Python 3.12)
- `chronic-risk-frontend/` — React 19 + Vite SPA (the educational UI)

There is no top-level git repo; treat each subfolder as an independent repository when running `git` commands.

## Common commands

### Backend (`chronic-risk-backend/`)

```powershell
# Install (use the bundled venv or create a new one)
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the API (loads models, builds SHAP explainers, opens SQLite DB)
python app.py            # serves on http://0.0.0.0:8000, debug mode

# Full data → model pipeline (run in order)
python prepare_datasets.py        # data_raw/  -> data_processed/{disease}_dataset.csv
python curate_and_synthesize.py   # data_processed/ -> data_curated/{disease}/ (train/test + CTGAN/TVAE synthetic + data dictionary)
python train_models.py            # data_curated/ -> models/{disease}_pipeline.pkl + _features.json + _metrics.json

# Useful flags on curate_and_synthesize.py
python curate_and_synthesize.py --model ctgan --synth_multiplier 1.0 --balance --epochs 50 --only diabetes,hipertension

# Production-style run (runtime.txt pins Python 3.12.8, gunicorn is in requirements)
gunicorn app:app
```

### Frontend (`chronic-risk-frontend/`)

```powershell
npm install
npm run dev        # Vite dev server
npm run build      # production build to dist/
npm run preview    # preview built bundle
npm run lint       # eslint .
```

The frontend reads `VITE_API_URL` (see `src/services/api.js`) and falls back to `http://localhost:8000`. `vercel.json` configures SPA rewrites for Vercel hosting.

## Architecture

### Backend — model serving + dynamic clinical rules

`app.py` is the single Flask entrypoint. At import time it runs `_load_all()` then `init_db()`, so the module has side effects: it loads all four sklearn pipelines from `models/`, builds a SHAP `LinearExplainer` per disease against a sampled background from `data_curated/<disease>/<disease>_train.csv` (falling back to `data_processed/<disease>_dataset.csv`), and creates the SQLite `predictions` table in `medical_history.db`.

Four diseases are supported as a closed set, declared in the `FILES` dict: `diabetes`, `hipertension`, `obesidad`, `cardiovascular`. Each has a `_pipeline.pkl` (sklearn `Pipeline` with steps named `scaler` + `clf`), a `_features.json` (ordered feature list), and a `_metrics.json`. New diseases must be added to `FILES` **and** to `DATASETS` in `train_models.py` to be picked up.

Endpoints (all under `/`, CORS open):
- `GET /health` — liveness
- `GET /metrics/<disease>` — returns the saved `_metrics.json` (includes both train and test AUC/classification reports)
- `GET /config/<disease>` — feature list + ranges + parsed categorical options (derived from one-hot column names like `gender_*`, `smoking_history_*`)
- `POST /predict/<disease>` — JSON in → probability + prediction + SHAP `top_features`
- `GET /synthetic/<disease>` — returns one random row from `data_curated/<disease>/<disease>_synthetic_ctgan*.csv` (prefers CTGAN, falls back to any `*_synthetic*.csv`, then to `data_processed/<disease>_dataset.csv`); the response includes `_source_type` = `"synthetic"` or `"real"`.

`/predict` has non-obvious behavior that must be preserved:
1. **Glucose alias normalization** — `glucose` and `blood_glucose_level` are mirrored into each other so the same payload works against datasets/models that use either name.
2. **Obesity short-circuit** — if `disease == "obesidad"` and the payload has any `bmi`, the model is bypassed entirely and a deterministic WHO rule (`BMI >= 30`) returns probability 0.99 or 0.01. The response sets `method: "clinical_rule_bmi"` and `top_features: []`. The ML model is only used when BMI is absent.
3. **Progressive clinical floors** — after the model probability is computed, per-disease rules (`diabetes`: glucose/HbA1c thresholds; `hipertension`: systolic BP bands) raise the probability via `max(prob, ...)`. They never lower it. The response distinguishes `raw_model_probability` (pre-rule) from `probability` (post-rule), and `explain_note` reminds the consumer that SHAP explains the *raw* model only.
4. **SHAP gender filter** — explanations omit any feature whose name starts with `gender_` before taking the top 5 by absolute SHAP value.
5. **Missing features** — anything not in the payload is filled with 0; the names of non-dummy missing features are returned in `missing_filled_as_zero`.

Every prediction is logged to the `predictions` table (`disease, input_data` as JSON text, `prediction`, `probability`, `timestamp`).

### Backend — data and training pipeline

`prepare_datasets.py` is a fan-in normalizer: it loads heterogeneous public CSVs from `data_raw/` (PIMA diabetes, an English diabetes-prediction set, a Mexican hypertension survey, an English hypertension set, the UCI obesity set, the Kaggle cardio set), routes each through a `is_*` detector to the matching `normalize_*` function, and concatenates the results into a single master frame using `COMMON_SCHEMA`. Then it does median imputation of physiological columns (`MEDICAL_DEFAULTS` is the last-resort fallback when an entire column is empty) and writes four per-disease CSVs into `data_processed/` with appropriate `target` re-derivation (diabetes keeps it; hipertensión = `hypertension`; obesidad = `bmi >= 30`; cardiovascular = `heart_disease OR target`).

`curate_and_synthesize.py` does stratified train/test split per disease, then trains an SDV synthesizer (CTGAN by default, TVAE optional) on the train split to emit `{disease}_synthetic_<model>_x<mul>_seed<seed>.csv`. Supports both SDV 1.x (`SingleTableMetadata`) and 0.x APIs. Also generates a `data_dictionary_{disease}.csv` + `.md` per disease using `COLUMN_DESCRIPTIONS`. Sanitization caps physiologically impossible upper bounds (`blood_pressure>300`, `glucose>500`, `bmi>80`) before fitting.

`train_models.py` trains a `StandardScaler(with_mean=False) → LogisticRegression(class_weight="balanced", max_iter=2000)` pipeline per disease against the curated split (falls back to a fresh split of `data_processed/` if curated files are missing). `get_features_for_disease()` removes leakage features per target (`hypertension` for the hipertensión model, `bmi` for obesidad, `heart_disease` for cardiovascular). The SHAP `LinearExplainer` in `app.py` depends on the pipeline keeping the exact step names `"scaler"` and `"clf"` — don't rename them.

### Frontend

React 19 SPA using react-router-dom v7. Bootstrap 5 + react-bootstrap for UI, recharts for charts, sweetalert2 for modals, axios for HTTP. Routes are declared in `src/App.jsx`: `/`, `/educacion`, `/simulacion`, `/metricas`, `/evaluacion`, `/aviso`, `/historial`.

- `src/services/api.js` — thin axios wrapper exposing `checkHealth`, `getConfig`, `getMetrics`, `predictRisk`, `getSyntheticCase`. Base URL is `VITE_API_URL` or `http://localhost:8000`.
- `src/services/historyService.js` — simulation history is **client-side only**, stored in `localStorage` under key `chronic_sim_history_v1`. Provides JSON/CSV export. The backend's SQLite `predictions` table is independent of this and is not exposed via any read endpoint.
- `src/utils/translations.js` — Spanish UI labels (`LABELS_ES` + `getLabel`); the app is Spanish-first.
- `src/pages/Simulacion.jsx` holds the prediction form and is the most complex page — it consumes `/config/<disease>`, `/predict/<disease>`, `/synthetic/<disease>`, applies the `CLINICAL_LIMITS` ranges to form inputs, and renders the SHAP `top_features` as a recharts bar.

## Conventions and gotchas

- The codebase is Spanish-first in user-facing strings, identifiers, and comments. Keep that style when editing (e.g., `hipertension`, `obesidad`).
- Disease keys are lowercase everywhere; routes lowercase the path segment before lookup. New diseases must be added consistently in `FILES` (app.py), `DATASETS` (train_models.py), and the frontend's `DISEASES` list (`Simulacion.jsx`).
- `medical_history.db` lives in the backend folder and is created on first run by `init_db()`. It is gitignored — don't commit it.
- The sklearn pipeline names `scaler` and `clf` are load-bearing for SHAP construction in `_build_shap_explainer`. Changing them silently breaks explainability without breaking prediction.
- `requirements.txt` in this repo is UTF-16-encoded (you'll see spaced characters when reading); install via pip still works, but editing it requires preserving the encoding or rewriting in UTF-8.
- `runtime.txt` pins `python-3.12.8` for deployment platforms (Heroku-style).
