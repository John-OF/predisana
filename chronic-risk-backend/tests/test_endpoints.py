# Tests de los endpoints publicos de solo lectura: health, metrics y config por modo,
# y el laboratorio sintetico (synthetic/sample/distribution/synthetic_quality).
import pytest

import modos as M

DISEASES = list(M.ENFERMEDADES)
MODOS = ["simplificado", "completo"]


def test_health(client):
    assert client.get("/health").status_code == 200


# ---------- /metrics ----------

@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("disease", DISEASES)
def test_metrics_por_enfermedad_y_modo(client, disease, modo):
    r = client.get(f"/metrics/{disease}?mode={modo}")
    assert r.status_code == 200
    m = r.get_json()
    assert m["modo"] == modo
    assert 0.5 < (m.get("auc") or m.get("auc_test")) <= 1.0
    assert m["leaderboard"], "leaderboard vacio"
    # La curva de fiabilidad que dibuja Metricas.jsx.
    assert "brier_calibrated" in m["calibration"]

def test_metrics_sin_modo_es_el_simplificado(client):
    assert client.get("/metrics/diabetes").get_json()["modo"] == "simplificado"

def test_metrics_enfermedad_desconocida(client):
    assert client.get("/metrics/obesidad").status_code == 404

def test_modo_desconocido_es_400(client):
    for ruta in ("/metrics/diabetes", "/config/diabetes"):
        assert client.get(f"{ruta}?mode=avanzado").status_code == 400


# ---------- /config ----------

def test_config_simplificado_pide_peso_y_talla_no_el_imc(client):
    """El simplificado se entrena con el IMC que sale del peso y la talla declarados:
    el formulario pide esos dos y la API calcula el IMC."""
    c = client.get("/config/diabetes?mode=simplificado").get_json()
    assert "bmi" in c["features"] and "bmi" not in c["inputs"]
    assert {"weight", "height"} <= set(c["inputs"])
    assert c["derived"] == {"bmi": ["weight", "height"]}
    assert c["categoricals"]["smoking_history"] == ["current", "former", "never"]

def test_config_completo_pide_el_imc_medido(client):
    c = client.get("/config/diabetes?mode=completo").get_json()
    assert "bmi" in c["inputs"] and c["derived"] == {}
    assert {"waist_circumference", "egfr", "albumin_creatinine_ratio"} <= set(c["inputs"])

@pytest.mark.parametrize("disease,definitorias", [
    ("diabetes", ["hba1c_level", "blood_glucose_level"]), ("hipertension", ["ap_hi", "ap_lo"])])
def test_lo_que_define_la_enfermedad_se_pide_pero_no_entra_al_modelo(client, disease, definitorias):
    """Se pide en el completo, pero lo interpreta la guia: con ello el modelo solo
    reaprenderia el umbral diagnostico."""
    c = client.get(f"/config/{disease}?mode=completo").get_json()
    assert c["defining_inputs"] == definitorias
    assert not set(definitorias) & set(c["features"])
    assert set(definitorias) <= set(c["optional_features"])

def test_config_avisa_de_lo_que_el_modelo_no_usa(client):
    c = client.get("/config/cardiovascular?mode=completo").get_json()
    assert {"ap_hi", "total_cholesterol"} <= set(c["not_used"])


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
    r = client.get("/distribution/diabetes?feature=hba1c_level")
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



@_pytest.mark.parametrize("enf", list(M.ENFERMEDADES))
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


@_pytest.mark.parametrize("enf", list(M.ENFERMEDADES))
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
