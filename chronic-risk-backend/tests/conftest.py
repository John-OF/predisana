# conftest.py — arranque de la suite.
# `app.py` tiene side effects al importar (_load_all + init_db), asi que TODO el
# entorno se prepara aqui ANTES de que cualquier test importe `app`:
#   - cwd = carpeta del backend (las rutas de modelos/datos son relativas)
#   - DATABASE_URL -> SQLite temporal (los tests NUNCA tocan medical_history.db)
#   - ADMIN_TOKEN de prueba (habilita los endpoints /admin/*)
import os
import sys
import tempfile

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

_TMP_DB_DIR = tempfile.mkdtemp(prefix="predisana_tests_")
os.environ["DATABASE_URL"] = (
    "sqlite:///" + os.path.join(_TMP_DB_DIR, "test_medical_history.db").replace(os.sep, "/")
)

TEST_ADMIN_TOKEN = "token-de-pruebas-pytest"
os.environ["ADMIN_TOKEN"] = TEST_ADMIN_TOKEN


@pytest.fixture(scope="session")
def app_module():
    """El modulo `app` ya inicializado (modelos + SHAP + BD temporal)."""
    import app
    return app


@pytest.fixture(scope="session")
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture()
def admin_headers():
    return {"X-Admin-Token": TEST_ADMIN_TOKEN}


# Perfil base RESPONDIBLE de diabetes (sin glucosa), compartido entre tests.
@pytest.fixture()
def perfil_diabetes():
    return {
        "age": 55, "bmi": 31, "hypertension": 1, "heart_disease": 0,
        "gender_Male": 1, "gender_Female": 0,
        "smoking_history_never": 1, "smoking_history_current": 0,
        "smoking_history_former": 0,
    }
