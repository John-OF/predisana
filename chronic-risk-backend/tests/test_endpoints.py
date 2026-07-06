# Tests de los endpoints publicos de solo lectura: health, metrics (incl. la
# variante hibrida), config y el laboratorio sintetico (synthetic/sample/
# distribution/synthetic_quality).
import pytest

DISEASES = ["diabetes", "hipertension", "cardiovascular"]


def test_health(client):
    assert client.get("/health").status_code == 200


# ---------- /metrics ----------

@pytest.mark.parametrize("disease", DISEASES)
def test_metrics_por_enfermedad(client, disease):
    r = client.get(f"/metrics/{disease}")
    assert r.status_code == 200
    m = r.get_json()
    assert 0.5 < (m.get("auc") or m.get("auc_test")) <= 1.0
    assert m["leaderboard"], "leaderboard vacio"
    # La curva de fiabilidad que dibuja Metricas.jsx.
    assert "brier_calibrated" in m["calibration"]

def test_metrics_sirve_variante_glucosa(client):
    """El toggle de /metricas pide la variante hibrida por su clave extra."""
    r = client.get("/metrics/diabetes_glucosa")
    assert r.status_code == 200
    m = r.get_json()
    assert "blood_glucose_level" in m["features"]

def test_metrics_enfermedad_desconocida(client):
    assert client.get("/metrics/obesidad").status_code == 404


# ---------- /config ----------

def test_config_diabetes_glucosa_es_opcional(client):
    """Contrato del hibrido: la glucosa NO es feature del modelo base pero SI
    se ofrece como opcional (asi la pinta el simulador)."""
    r = client.get("/config/diabetes")
    assert r.status_code == 200
    c = r.get_json()
    assert "blood_glucose_level" not in c["features"]
    assert "blood_glucose_level" in c["optional_features"]
    assert c["categoricals"]["gender"], "sin opciones de genero"

@pytest.mark.parametrize("disease", ["hipertension", "cardiovascular"])
def test_config_sin_opcionales_en_otras(client, disease):
    r = client.get(f"/config/{disease}")
    assert r.status_code == 200
    assert r.get_json()["optional_features"] == []


# ---------- laboratorio sintetico ----------

@pytest.mark.parametrize("disease", DISEASES)
def test_synthetic_devuelve_ficha(client, disease):
    r = client.get(f"/synthetic/{disease}")
    assert r.status_code == 200
    assert r.get_json()["_source_type"] in ("synthetic", "real")

def test_synthetic_enfermedad_desconocida(client):
    assert client.get("/synthetic/obesidad").status_code == 404

def test_sample_real(client):
    r = client.get("/sample/diabetes?source=real")
    assert r.status_code == 200
    assert r.get_json()["_source_type"] == "real"

def test_distribution_comparada(client):
    r = client.get("/distribution/diabetes?feature=blood_glucose_level")
    assert r.status_code == 200
    d = r.get_json()
    assert d["bins"], "sin bins"
    # Cada serie esta normalizada a % (compara la forma, no el tamano de muestra).
    assert sum(b["real"] for b in d["bins"]) == pytest.approx(100, abs=1.0)
    assert sum(b["synthetic"] for b in d["bins"]) == pytest.approx(100, abs=1.0)

def test_distribution_feature_invalida(client):
    assert client.get("/distribution/diabetes?feature=no_existe").status_code == 400

def test_synthetic_quality(client):
    r = client.get("/synthetic_quality/diabetes")
    assert r.status_code == 200
    q = r.get_json()
    assert q["overall"] is not None and 0.0 < q["overall"] <= 1.0
    assert q["corr"]["features"], "sin matriz de correlaciones"
