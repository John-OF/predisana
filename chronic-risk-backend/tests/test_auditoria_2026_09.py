"""Auditoria 2026-09 — hallazgos de severidad baja, y de la media /health + rutas.

Cada test fija el comportamiento CORREGIDO. Los dos de severidad alta (JSON invalido
con datos clinicos no finitos) viven en test_auditoria.py, junto a AUD-2; la fuerza
bruta del token repartida entre endpoints, en test_seguridad_cors_ratelimit.py. Desde
la v2 los perfiles son del modo simplificado (peso y talla en vez del IMC).
"""
import glob
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import modos as M

CLAVES = sorted(f"{d}_{m}" for d in M.ENFERMEDADES for m in M.MODOS)

# Lleva los cuatro diagnosticos: cada modelo lee los suyos y la API ignora el resto.
SIMPLE = {
    "age": 45, "weight": 75, "height": 175, "diabetes": 0, "hypertension": 0,
    "heart_disease": 0, "high_cholesterol": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
COMPLETO = {
    **{k: v for k, v in SIMPLE.items() if k not in ("weight", "height")},
    "bmi": 24.5, "waist_circumference": 88, "ap_hi": 122, "ap_lo": 78,
    "total_cholesterol": 190, "hdl_cholesterol": 52, "hba1c_level": 5.4, "egfr": 95,
    "albumin_creatinine_ratio": 7,
}


# ---------- Limites fisicos de entrada ----------

@pytest.mark.parametrize("enfermedad,modo,campo,valor", [
    ("hipertension", "simplificado", "age", -30),
    ("hipertension", "simplificado", "weight", 10),
    ("hipertension", "simplificado", "height", 300),
    ("hipertension", "simplificado", "bmi", 900),
    ("hipertension", "simplificado", "ap_hi", -40),
    ("hipertension", "simplificado", "ap_hi", 400),
    ("cardiovascular", "simplificado", "gender_Male", 7),
    ("diabetes", "simplificado", "blood_glucose_level", 1000),
    ("diabetes", "simplificado", "hba1c_level", 50),
    ("cardiovascular", "completo", "ap_lo", 500),
    ("cardiovascular", "completo", "egfr", 0),
    ("diabetes", "completo", "albumin_creatinine_ratio", 30000),
    ("hipertension", "completo", "total_cholesterol", 900),
])
def test_valor_fisicamente_imposible_da_400(client, enfermedad, modo, campo, valor):
    """Antes: 200, probabilidad 1.0 y una fila basura en la BD del admin."""
    base = SIMPLE if modo == "simplificado" else COMPLETO
    r = client.post(f"/predict/{enfermedad}?mode={modo}", json={**base, campo: valor})
    assert r.status_code == 400
    assert campo in r.get_json()["error"] and "fuera de rango" in r.get_json()["error"]


@pytest.mark.parametrize("edad", [18, 100])
def test_los_extremos_del_limite_se_aceptan(client, edad):
    assert client.post("/predict/hipertension", json={**SIMPLE, "age": edad}).status_code == 200


def test_whatif_no_barre_fuera_de_los_limites(client):
    r = client.post("/whatif/hipertension",
                    json={"base": SIMPLE, "feature": "age", "min": 0, "max": 200, "steps": 5})
    assert r.status_code == 400


@pytest.mark.parametrize("modo", M.MODOS)
@pytest.mark.parametrize("enfermedad", M.ENFERMEDADES)
def test_todo_campo_del_formulario_tiene_limite(client, app_module, enfermedad, modo):
    """Si el esquema cambia (como al migrar a NHANES), una feature nueva sin limite
    quedaria sin validar y sin min/max en el formulario."""
    c = client.get(f"/config/{enfermedad}?mode={modo}").get_json()
    for f in c["features"] + c["inputs"] + c["optional_features"]:
        if f not in c["categoricals"]:
            assert app_module._input_limits(f) is not None, f"{enfermedad}/{modo}: '{f}' sin limite"


@pytest.mark.parametrize("enfermedad", M.ENFERMEDADES)
def test_los_limites_no_rechazan_ningun_dato_real_ni_sintetico(app_module, enfermedad):
    """Un limite es un tope FISICO, no el rango entrenado: ninguna fila real ni
    sintetica puede quedar fuera (p.ej. hay adultos reales de 27,9 kg en NHANES). El
    IMC declarado es el `bmi` del simplificado: se juzga con el limite del IMC."""
    como_feature = {columna: f for (_, f), columna in M.FUENTE.items()}
    for path in glob.glob(os.path.join("data_curated", enfermedad, "*.csv")):
        df = pd.read_csv(path)
        for col in df.columns:
            limites = app_module._input_limits(como_feature.get(col, col))
            if limites is None:
                continue
            serie = df[col].dropna()
            fuera = serie[(serie < limites[0]) | (serie > limites[1])]
            assert fuera.empty, f"{os.path.basename(path)}:{col} {fuera.tolist()[:5]} fuera de {limites}"


@pytest.mark.parametrize("modo", M.MODOS)
@pytest.mark.parametrize("enfermedad", M.ENFERMEDADES)
def test_config_sirve_los_mismos_limites_que_valida(client, app_module, enfermedad, modo):
    """/config.ranges era un dict fijo que el front no leia y que contradecia sus
    propios limites (presion 60-130 frente a 50-300)."""
    c = client.get(f"/config/{enfermedad}?mode={modo}").get_json()
    for f in c["features"] + c["optional_features"]:
        assert c["ranges"][f] == app_module._input_limits(f), f


# ---------- HbA1c y glucosa: los lee la guia ADA, no el modelo ----------

@pytest.mark.parametrize("modo", M.MODOS)
def test_hba1c_y_glucosa_son_datos_clinicos_de_diabetes(client, modo):
    """Definen la diabetes (objetivo v2: diagnosticada, HbA1c >= 6,5% o glucosa en
    ayunas >= 126): con ellas el modelo solo reaprenderia el umbral. Se pueden aportar
    en los dos modos y las interpreta la capa clinica."""
    c = client.get(f"/config/diabetes?mode={modo}").get_json()
    assert {"hba1c_level", "blood_glucose_level"} <= set(c["clinical_inputs"])
    assert c["defining_inputs"] == ["hba1c_level", "blood_glucose_level"]
    assert not {"hba1c_level", "blood_glucose_level"} & set(c["features"])


@pytest.mark.parametrize("campo,valor,indicador", [("hba1c_level", 9.5, "hba1c"),
                                                   ("blood_glucose_level", 160, "glucose")])
def test_el_dato_de_la_guia_dispara_su_indicador_sin_tocar_la_probabilidad(
        client, perfil_diabetes, campo, valor, indicador):
    sin = client.post("/predict/diabetes", json=perfil_diabetes).get_json()
    con = client.post("/predict/diabetes", json={**perfil_diabetes, campo: valor}).get_json()
    assert [f["category"] for f in con["clinical_flags"] if f["indicator"] == indicador] == ["diabetes"]
    assert con["probability"] == sin["probability"]


# ---------- Admin: limit <= 0 no se salta el tope ----------

@pytest.mark.parametrize("limit", ["-1", "0"])
def test_admin_limit_no_positivo_usa_el_de_por_defecto(client, app_module, admin_headers, limit):
    """SQLite lee LIMIT -1 como "sin limite": devolvia la tabla entera."""
    for _ in range(60):
        app_module.log_prediction_to_db("diabetes", {"age": 50}, 0, 0.1)
    r = client.get(f"/admin/predictions?limit={limit}", headers=admin_headers)
    assert r.status_code == 200
    assert r.get_json()["count"] == 50


# ---------- Solo se sirven enfermedades ----------

@pytest.mark.parametrize("ruta", ["diabetes_glucosa", "diabetes_completo", "hipertension_simplificado"])
def test_solo_se_sirven_enfermedades(client, ruta):
    """La variante con glucosa de la v1 ya no existe, y la clave interna de cada modelo
    (<enfermedad>_<modo>) no es una ruta: el modo va en ?mode=."""
    assert client.get(f"/config/{ruta}").status_code == 404
    assert client.get(f"/metrics/{ruta}").status_code == 404
    assert client.post(f"/predict/{ruta}", json=SIMPLE).status_code == 404
    assert client.post(f"/whatif/{ruta}", json={"base": SIMPLE, "feature": "age",
                                                "min": 20, "max": 80, "steps": 3}).status_code == 404


# ---------- Coste de los endpoints publicos ----------

def test_csv_del_laboratorio_se_leen_una_sola_vez(client, app_module, monkeypatch):
    """/sample, /synthetic y /distribution releian el CSV entero en cada peticion."""
    app_module._read_csv_cached.cache_clear()
    lecturas = []
    original = pd.read_csv
    monkeypatch.setattr(app_module.pd, "read_csv",
                        lambda path, *a, **kw: lecturas.append(path) or original(path, *a, **kw))
    for _ in range(5):
        assert client.get("/sample/cardiovascular?source=real").status_code == 200
        assert client.get("/sample/cardiovascular?source=synthetic").status_code == 200
        assert client.get("/distribution/cardiovascular?feature=age").status_code == 200
    assert len(lecturas) == 2, lecturas   # train real + sintetico, una vez cada uno
    app_module._read_csv_cached.cache_clear()


def test_health_no_consulta_la_bd_en_cada_peticion(client, app_module, monkeypatch):
    """/health esta exento del rate limiting (AUD-4): un bucle contra el no puede
    traducirse en una consulta a la BD por peticion."""
    app_module._health_db["checked_at"] = None
    conexiones = []
    original = app_module.engine.connect
    monkeypatch.setattr(app_module.engine, "connect",
                        lambda *a, **kw: conexiones.append(1) or original(*a, **kw))
    codigos = {client.get("/health").status_code for _ in range(20)}
    assert codigos == {200}
    assert len(conexiones) == 1
    app_module._health_db["checked_at"] = None


# ---------- /health sin modelos (severidad media) ----------

def test_health_sin_modelos_responde_503(client, app_module, monkeypatch):
    """Antes: 200 "ok" con models_loaded vacio. El deploy se daba por sano mientras
    todos los /predict respondian 500."""
    monkeypatch.setattr(app_module, "MODELS", {})
    r = client.get("/health")
    assert r.status_code == 503
    d = r.get_json()
    assert d["status"] == "degraded"
    assert d["database_ok"] is True
    assert d["models_missing"] == CLAVES


def test_health_sin_un_modo_responde_503(client, app_module, monkeypatch):
    """Sin el modelo de un modo, /predict de ese modo responde 500 aunque el otro
    funcione: tambien es un deploy roto."""
    monkeypatch.delitem(app_module.MODELS, "diabetes_completo")
    r = client.get("/health")
    assert r.status_code == 503
    assert r.get_json()["models_missing"] == ["diabetes_completo"]


# ---------- Rutas ancladas a la carpeta del backend (severidad media) ----------

_ARRANQUE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import app
c = app.app.test_client()
salud = c.get("/health")
print(json.dumps({
    "health": [salud.status_code, salud.get_json()],
    "shap": sorted(app.EXPLAINERS),
    "support": sorted(app.SUPPORT),
    "prevalence": sorted(app.PREVALENCE),
    "sample": c.get("/sample/diabetes?source=real").status_code,
    "synthetic": c.get("/sample/diabetes?source=synthetic").status_code,
    "distribution": c.get("/distribution/diabetes?feature=age").status_code,
    "quality": c.get("/synthetic_quality/diabetes").status_code,
}))
"""


def test_la_app_arranca_desde_otra_carpeta(app_module, tmp_path):
    """Las rutas eran relativas al directorio de trabajo. Arrancada desde otra carpeta
    (la raiz del repo, o un deploy sin `--chdir`), la app levantaba sin ningun
    modelo ni datos, /predict daba 500 y /health decia "ok"."""
    backend = os.path.dirname(os.path.abspath(app_module.__file__))
    env = {**os.environ,
           "DATABASE_URL": "sqlite:///" + str(tmp_path / "otra.db").replace(os.sep, "/")}
    salida = subprocess.run(
        [sys.executable, "-c", _ARRANQUE, backend], cwd=tmp_path, env=env,
        capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert salida.returncode == 0, salida.stderr[-2000:]
    r = json.loads(salida.stdout.strip().splitlines()[-1])
    codigo, salud = r["health"]
    assert codigo == 200 and salud["status"] == "ok"
    assert salud["models_missing"] == []
    # SHAP y la cobertura leen el train de data_curated.
    assert salud["models_loaded"] == r["shap"] == r["support"] == CLAVES
    assert r["prevalence"] == sorted(M.ENFERMEDADES)
    assert (r["sample"], r["synthetic"], r["distribution"], r["quality"]) == (200, 200, 200, 200)


# ---------- Tope de edad de NHANES (80 = "80 o mas") ----------

def test_edad_sobre_el_tope_de_nhanes_no_dice_que_no_vio_casos(client):
    r = client.post("/predict/hipertension", json={**SIMPLE, "age": 85}).get_json()
    [aviso] = [a for a in r["support_warnings"] if a["feature"] == "age"]
    assert aviso["topcoded"] == 80
    assert "no vio ningún caso" not in aviso["detail"]
    assert "figura como 80" in aviso["detail"]


@pytest.mark.parametrize("enfermedad", M.ENFERMEDADES)
def test_whatif_y_config_exponen_el_tope(client, enfermedad):
    """Desde la v2 cardiovascular tambien sale de NHANES: las tres tienen el tope."""
    wi = client.post(f"/whatif/{enfermedad}",
                     json={"base": SIMPLE, "feature": "age", "min": 18, "max": 90, "steps": 3}).get_json()
    assert wi["topcoded_at"] == 80
    assert client.get(f"/config/{enfermedad}").get_json()["topcoded"] == {"age": 80}
