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

def test_config_hipertension_ofrece_la_presion_como_opcional(client):
    """AUD-1: la presion NO es feature del modelo (seria un umbral disfrazado),
    pero se ofrece como dato opcional para la capa clinica ACC/AHA."""
    c = client.get("/config/hipertension").get_json()
    assert "blood_pressure" not in c["features"]
    assert c["optional_features"] == ["blood_pressure"]

def test_config_cardiovascular_sin_opcionales(client):
    assert client.get("/config/cardiovascular").get_json()["optional_features"] == []


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


# ---------- TSTR y privacidad (DCR) en el informe de calidad ----------
# El laboratorio solo enseñaba FIDELIDAD, que es la pregunta facil. Estas son las dos
# que decide un revisor: si el sintetico SIRVE y si NO copia a nadie.
import pytest as _pytest


@_pytest.mark.parametrize("enf", ["diabetes", "hipertension", "cardiovascular"])
def test_calidad_incluye_tstr(client, enf):
    d = client.get(f"/synthetic_quality/{enf}").get_json()
    tstr = d["tstr"]
    assert tstr["n_test_real"] > 0 and tstr["n_train_synth"] > 0
    for m in tstr["models"]:
        # Un sintetico util tiene que quedarse cerca del real, no empatarlo: si el
        # ratio se fuera por encima de ~1.05 seria sospechoso (el sintetico no puede
        # saber mas que los datos de los que salio).
        assert 0.5 < m["ratio"] <= 1.05, (enf, m)
        assert 0.5 < m["tstr_auc"] < 1.0


@_pytest.mark.parametrize("enf", ["diabetes", "hipertension", "cardiovascular"])
def test_calidad_incluye_privacidad(client, enf):
    p = client.get(f"/synthetic_quality/{enf}").get_json()["privacy"]
    # La referencia no es cero: se compara con lo que dista el propio test real.
    assert p["median_real_test"] > 0
    assert p["ratio"] >= 1.0, f"{enf}: el sintetico esta MAS cerca del train que el test real"
    # Copias exactas: se toleran si el dato es grueso, pero nunca mas que entre reales.
    tasa_synth = p["exact_copies"] / p["n_synthetic"]
    tasa_real = p["exact_copies_real_test"] / p["n_real_test"]
    assert tasa_synth <= max(tasa_real, 0.001), (enf, tasa_synth, tasa_real)


def test_el_tstr_usa_el_mismo_algoritmo_en_las_dos_ramas(client):
    """Si cada rama usara un modelo distinto, la comparacion mediria el algoritmo y no
    los datos, que es justo lo que TSTR quiere aislar."""
    modelos = client.get("/synthetic_quality/diabetes").get_json()["tstr"]["models"]
    nombres = [m["model"] for m in modelos]
    assert len(nombres) == len(set(nombres))
    for m in modelos:
        assert m["trtr_auc"] is not None and m["tstr_auc"] is not None
