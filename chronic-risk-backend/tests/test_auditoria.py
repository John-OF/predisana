# Regresiones de la auditoria 2026-08-20 (AUD-1, 2, 3, 5, 6, 7, 8, 9, 14, 15 y 18).
# Cada test fija el comportamiento CORREGIDO para que no vuelva a colarse. Desde la v2
# (NHANES 2017-2023, dos modos por enfermedad) los perfiles son del modo simplificado:
# peso y talla en vez del IMC.
import json
import math

import pandas as pd
import pytest

import modos as M

CLAVES = [(d, m) for d in M.ENFERMEDADES for m in M.MODOS]


# ---------- AUD-2: NaN nunca sale al JSON ----------

def test_json_safe_convierte_nan_a_none(app_module):
    assert app_module._json_safe(float("nan")) is None
    assert app_module._json_safe(float("inf")) is None
    assert app_module._json_safe(3.14159) == 3.14
    assert app_module._json_safe(None) is None


@pytest.mark.parametrize("ruta", ["/sample/diabetes?source=real",
                                  "/sample/diabetes?source=synthetic",
                                  "/synthetic/diabetes"])
def test_fichas_de_paciente_son_json_valido(client, ruta):
    """El train real de NHANES trae labs ausentes: antes se serializaban como el
    literal `NaN`, que rompe el JSON.parse del navegador (~10% de las fichas)."""
    for _ in range(40):
        r = client.get(ruta)
        assert r.status_code == 200
        crudo = r.get_data(as_text=True)
        assert "NaN" not in crudo and "Infinity" not in crudo, crudo[:200]
        ficha = json.loads(crudo)  # parseo estricto, como el navegador
        for k, v in ficha.items():
            assert not (isinstance(v, float) and not math.isfinite(v)), k


# Mismo fallo por otra via (auditoria 2026-09): los datos clinicos que NO entran al
# modelo (la presion en hipertension, la glucosa en cardiovascular) se leian con
# float() a secas, y un inf acababa como `Infinity` en clinical_flags.
_HTA = {
    "age": 45, "weight": 75, "height": 175, "diabetes": 0, "heart_disease": 0,
    "high_cholesterol": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
_CARDIO = {
    "age": 50, "weight": 68, "height": 163, "diabetes": 0, "hypertension": 0,
    "high_cholesterol": 0, "gender_Male": 0, "gender_Female": 1,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}


@pytest.mark.parametrize("enfermedad,base,campo", [
    ("hipertension", _HTA, "ap_hi"),
    ("hipertension", _HTA, "ap_lo"),
    ("hipertension", _HTA, "blood_glucose_level"),
    ("cardiovascular", _CARDIO, "blood_glucose_level"),
    ("cardiovascular", _CARDIO, "hba1c_level"),
])
@pytest.mark.parametrize("valor", [float("inf"), "inf", "NaN", "ciento cuarenta"])
def test_dato_clinico_opcional_invalido_da_400(client, enfermedad, base, campo, valor):
    r = client.post(f"/predict/{enfermedad}", json={**base, campo: valor})
    assert r.status_code == 400
    assert campo in r.get_json()["error"]


def test_dato_clinico_opcional_valido_sigue_funcionando(client):
    """El arreglo no puede romper el caso normal: una sistolica de 150 sigue dando su
    indicador ACC/AHA, y la respuesta es JSON estricto."""
    r = client.post("/predict/hipertension", json={**_HTA, "ap_hi": 150})
    assert r.status_code == 200
    crudo = r.get_data(as_text=True)
    assert "Infinity" not in crudo and "NaN" not in crudo
    flags = json.loads(crudo)["clinical_flags"]
    assert [f["category"] for f in flags] == ["hipertension_grado_2"]


def test_dato_clinico_opcional_vacio_es_ausente(client):
    """Un input opcional borrado en el form manda "": es ausente, no un 400."""
    r = client.post("/predict/hipertension", json={**_HTA, "ap_hi": ""})
    assert r.status_code == 200
    assert r.get_json()["clinical_flags"] == []


# ---------- AUD-3: entrada invalida -> 400, enfermedad inexistente -> 404 ----------

@pytest.mark.parametrize("valor", ["cuarenta", [1, 2, 3], {"a": 1}, "NaN", float("inf")])
def test_predict_valor_no_numerico_da_400(client, perfil_diabetes, valor):
    r = client.post("/predict/diabetes", json={**perfil_diabetes, "age": valor})
    assert r.status_code == 400
    assert "age" in r.get_json()["error"]

@pytest.mark.parametrize("campo,se_nombra", [("age", "age"), ("weight", "weight y height")])
def test_predict_campo_vacio_se_trata_como_ausente(client, perfil_diabetes, campo, se_nombra):
    """Un input borrado en el form manda "": es ausente, no un 500. Desde la v2 lo que
    el modelo usa es obligatorio (la v1 lo rellenaba con 0, y un IMC de 0 no es un
    paciente): falta -> 400 que dice que falta."""
    r = client.post("/predict/diabetes", json={**perfil_diabetes, campo: ""})
    assert r.status_code == 400
    error = r.get_json()["error"]
    assert "faltan datos" in error and se_nombra in error

def test_predict_cuerpo_no_objeto_da_400(client):
    assert client.post("/predict/diabetes", json=[1, 2, 3]).status_code == 400

def test_predict_enfermedad_inexistente_da_404(client):
    r = client.post("/predict/obesidad", json={"age": 40})
    assert r.status_code == 404
    assert r.get_json()["error"] == "unknown disease"

def test_whatif_feature_desconocida_da_400(client, perfil_diabetes):
    r = client.post("/whatif/diabetes", json={"feature": "no_existe", "min": 0,
                                              "max": 10, "base": perfil_diabetes})
    assert r.status_code == 400

def test_whatif_base_no_objeto_da_400(client):
    r = client.post("/whatif/diabetes", json={"feature": "age", "min": 20,
                                              "max": 80, "base": [1, 2]})
    assert r.status_code == 400

def test_whatif_valor_no_numerico_en_base_da_400(client, perfil_diabetes):
    r = client.post("/whatif/diabetes", json={"feature": "age", "min": 20, "max": 80,
                                              "base": {**perfil_diabetes, "weight": "gordo"}})
    assert r.status_code == 400


# ---------- AUD-8: session_id saneado ----------

def test_session_id_se_trunca_y_sanea(client, admin_headers, perfil_diabetes):
    """El header lo controla el cliente. La columna es VARCHAR(64): en Postgres un
    id largo haria fallar el INSERT en silencio."""
    r = client.post("/predict/diabetes", json=perfil_diabetes,
                    headers={"X-Session-Id": "S" * 500})
    assert r.status_code == 200
    ultimo = client.get("/admin/predictions?limit=1",
                        headers=admin_headers).get_json()["items"][0]
    assert len(ultimo["session_id"]) == 64

def test_session_id_uuid_normal_se_conserva(client, admin_headers, perfil_diabetes):
    sesion = "3cc45d85-9117-4a4f-93cc-bfb51ca9265d"
    client.post("/predict/diabetes", json=perfil_diabetes,
                headers={"X-Session-Id": sesion})
    ultimo = client.get("/admin/predictions?limit=1",
                        headers=admin_headers).get_json()["items"][0]
    assert ultimo["session_id"] == sesion


# ---------- AUD-6: el CSV del admin no ejecuta formulas ----------

def test_export_csv_escapa_formulas(client, admin_headers, perfil_diabetes):
    """Un campo de texto controlado por el cliente no puede empezar por '=' en el
    CSV (Excel/Sheets lo ejecutaria)."""
    client.post("/predict/diabetes", json=perfil_diabetes,
                headers={"X-Session-Id": "=cmd|'/c calc'!A1"})
    csv_txt = client.get("/admin/export.csv", headers=admin_headers).get_data(as_text=True)
    for linea in csv_txt.splitlines()[1:]:
        for celda in linea.split(","):
            celda = celda.strip('"')
            assert not celda.startswith(("=", "+", "@")), celda


def test_csv_safe_antepone_comilla(app_module):
    assert app_module._csv_safe("=1+1").startswith("'")
    assert app_module._csv_safe("@SUM(A1)").startswith("'")
    assert app_module._csv_safe("diabetes") == "diabetes"


# ---------- AUD-18: /health comprueba la BD ----------

def test_health_reporta_estado_real_de_la_bd(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "ok"
    assert d["database_ok"] is True
    assert d["database"] == "sqlite"          # el motor real, no un string fijo
    assert d["models_loaded"] == sorted(f"{e}_{m}" for e, m in CLAVES)
    assert d["models_missing"] == []


# ---------- AUD-7: solo se persiste lo que el modelo usa ----------

def test_input_data_no_guarda_claves_arbitrarias(client, admin_headers, perfil_diabetes):
    """El payload lo controla el cliente: guardarlo entero engordaba la BD con
    texto libre que ademas acaba en el CSV del admin."""
    r = client.post("/predict/diabetes",
                    json={**perfil_diabetes, "basura": "x" * 500, "__proto__": "y"})
    assert r.status_code == 200
    guardado = client.get("/admin/predictions?limit=1",
                          headers=admin_headers).get_json()["items"][0]["input_data"]
    assert "basura" not in guardado and "__proto__" not in guardado
    # El peso y la talla que dio, y el IMC que sale de ellos (92 / 1,72^2).
    assert (guardado["age"], guardado["weight"], guardado["height"]) == (55, 92, 172)
    assert guardado["bmi"] == 31.1

def test_input_data_conserva_la_glucosa_aunque_no_sea_del_modelo_base(client, admin_headers,
                                                                     perfil_diabetes):
    """La glucosa alimenta la capa clinica ADA: debe seguir en el log."""
    client.post("/predict/diabetes", json={**perfil_diabetes, "blood_glucose_level": 170})
    guardado = client.get("/admin/predictions?limit=1",
                          headers=admin_headers).get_json()["items"][0]["input_data"]
    assert guardado["blood_glucose_level"] == 170


# ---------- AUD-9: tope de tamano del cuerpo ----------

@pytest.mark.parametrize("ruta,cuerpo", [
    ("/predict/diabetes", {"age": 40}),
    ("/whatif/diabetes", {"feature": "age", "min": 20, "max": 80, "base": {"age": 40}}),
])
def test_cuerpo_gigante_da_413(client, ruta, cuerpo):
    r = client.post(ruta, json={**cuerpo, "relleno": "x" * 300_000})
    assert r.status_code == 413
    assert r.get_json()["error"]

def test_cuerpo_normal_no_se_ve_afectado(client, perfil_diabetes):
    assert client.post("/predict/diabetes", json=perfil_diabetes).status_code == 200


# ---------- AUD-1: hipertension migrada a NHANES ----------

# 169 cm: con 80 kg, IMC 28.
PERFIL_HTA = {
    "age": 50, "weight": 80, "height": 169, "diabetes": 0, "heart_disease": 0,
    "high_cholesterol": 0, "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
}
PERFIL_HTA_COMPLETO = {
    **{k: v for k, v in PERFIL_HTA.items() if k not in ("weight", "height")},
    "bmi": 28, "waist_circumference": 97, "total_cholesterol": 195, "hdl_cholesterol": 48,
    "hba1c_level": 5.5, "egfr": 92, "albumin_creatinine_ratio": 9,
}


def _riesgo_hta(client, **cambios):
    r = client.post("/predict/hipertension", json={**PERFIL_HTA, **cambios})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["probability"]


def test_hipertension_el_riesgo_crece_con_la_edad(client):
    """El modelo viejo (target-formula de ENSANUT) daba 100% a los 20 años y 38% a
    los 80. Con NHANES la relacion es la clinica: a mas edad, mas riesgo."""
    riesgos = [_riesgo_hta(client, age=a) for a in (25, 40, 55, 70, 80)]
    assert riesgos == sorted(riesgos), riesgos
    assert riesgos[-1] > riesgos[0] * 2

def test_hipertension_el_riesgo_crece_con_el_peso(client):
    """A talla fija, de un IMC de 20 a uno de 40."""
    riesgos = [_riesgo_hta(client, weight=w) for w in (57, 77, 94, 114)]
    assert riesgos == sorted(riesgos), riesgos

def test_hipertension_no_satura_en_el_extremo_sano(client):
    """Un adulto joven y delgado no puede salir con un riesgo alto (el modelo viejo
    devolvia 100% con 25 años y presion 110)."""
    assert _riesgo_hta(client, age=25, weight=62, height=168) < 0.15

def test_hipertension_las_comorbilidades_suman(client):
    base = _riesgo_hta(client)
    assert _riesgo_hta(client, diabetes=1) > base
    assert _riesgo_hta(client, high_cholesterol=1) > base

@pytest.mark.parametrize("modo,perfil", [("simplificado", PERFIL_HTA),
                                         ("completo", PERFIL_HTA_COMPLETO)])
def test_presion_no_entra_al_modelo_pero_si_a_la_capa_clinica(client, modo, perfil):
    """La presion define la hipertension: con ella el modelo solo reaprenderia el
    umbral. La lee la guia ACC/AHA, sistolica y diastolica."""
    ruta = f"/predict/hipertension?mode={modo}"
    sin = client.post(ruta, json=perfil).get_json()
    con = client.post(ruta, json={**perfil, "ap_hi": 165, "ap_lo": 95}).get_json()
    assert con["probability"] == sin["probability"]      # la presion no mueve el modelo
    assert not sin["clinical_flags"]
    [flag] = [f for f in con["clinical_flags"] if f["indicator"] == "blood_pressure"]
    assert flag["source"] == "ACC/AHA" and flag["category"] == "hipertension_grado_2"


# ---------- AUD-14: la glucosa real cae en lo que acepta la API ----------
# La v1 traia glucosas de hasta 898 mg/dL (perfil bioquimico, sin ayuno) que acababan
# en el fondo de SHAP y en las fichas de 'caso real' del laboratorio. La v2 usa la de
# AYUNAS (hasta 561) y solo para definir el objetivo: no es variable de ningun modelo
# ni sale en las fichas. Que el resto de columnas reales caiga en los limites lo exige
# test_los_limites_no_rechazan_ningun_dato_real_ni_sintetico (auditoria 2026-09).

@pytest.mark.parametrize("ruta", [
    "data_processed/diabetes_dataset.csv",
    "data_curated/diabetes/diabetes_train.csv",
    "data_curated/diabetes/diabetes_test.csv",
])
def test_glucosa_real_dentro_de_los_limites(app_module, ruta):
    lo, hi = app_module.INPUT_LIMITS["blood_glucose_level"]
    g = pd.read_csv(ruta)["blood_glucose_level"].dropna()
    assert lo <= g.min() and g.max() <= hi, f"{ruta}: glucosa de {g.min()} a {g.max()}"


def test_la_glucosa_no_llega_al_fondo_de_shap_ni_a_las_fichas(app_module):
    for disease, modo in CLAVES:
        assert "blood_glucose_level" not in app_module.FEATURES[f"{disease}_{modo}"]
    for disease in M.ENFERMEDADES:
        assert "blood_glucose_level" not in M.columnas_laboratorio(disease)


# ---------- AUD-15: un solo reparto ----------
# `train_nhanes_diabetes.py` hacia su propio train_test_split del CSV completo
# mientras `data_curated/diabetes/*` salia de `curate_and_synthesize.py`: el test que
# reportaban las metricas no era el curado, y el 79% de sus filas de test estaban en
# el fondo de SHAP. Desde la v2 hay un solo reparto por enfermedad
# (train_models.repartir) y cada modo se queda con sus filas completas.

def _curado(disease, cual):
    return pd.read_csv(f"data_curated/{disease}/{disease}_{cual}.csv")


@pytest.mark.parametrize("disease", M.ENFERMEDADES)
def test_train_y_test_curados_son_disjuntos(disease):
    tr, te = _curado(disease, "train"), _curado(disease, "test")
    comunes = pd.merge(tr.round(6), te.round(6), how="inner")
    assert comunes.empty, f"{len(comunes)} filas compartidas entre train y test"


@pytest.mark.parametrize("disease,modo", CLAVES)
def test_las_metricas_se_reportan_sobre_el_test_curado(disease, modo):
    """El numero publicado en /metricas tiene que salir del MISMO test que sirve el
    laboratorio. En la v1, `diabetes_glucosa` reportaba sobre 1122 filas de otro reparto
    mientras el test curado tenia 1131."""
    te = _curado(disease, "test")
    columnas = [M.columna(modo, f) for f in M.features(disease, modo)]
    esperado = int(te[columnas].notna().all(axis=1).sum())
    with open(f"models/{disease}_{modo}_metrics.json", encoding="utf-8") as f:
        reportado = json.load(f)["report_test"]["macro avg"]["support"]
    assert int(reportado) == esperado


@pytest.mark.parametrize("disease,modo", CLAVES)
def test_la_app_usa_el_train_del_modelo(app_module, disease, modo):
    """El fondo de SHAP, la cobertura de datos (AUD-16) y las medianas salen de
    _train_modo: tienen que ser las filas con las que se entreno el modelo."""
    with open(f"models/{disease}_{modo}_metrics.json", encoding="utf-8") as f:
        n_train = json.load(f)["n_train"]
    assert len(app_module._train_modo(f"{disease}_{modo}")) == n_train


@pytest.mark.parametrize("disease", M.ENFERMEDADES)
def test_el_train_de_la_app_no_contiene_filas_del_test(app_module, disease):
    """En el completo (analiticas con decimales) dos personas distintas no comparten
    fila: si una coincide, es la misma, y SHAP se explicaria contra el test."""
    clave = f"{disease}_completo"
    feats = app_module.FEATURES[clave]
    te = _curado(disease, "test")
    test = pd.DataFrame({f: te[M.columna("completo", f)] for f in feats}).dropna()
    usado = app_module._train_modo(clave)[feats]
    comunes = pd.merge(usado.round(6), test.round(6), how="inner")
    assert comunes.empty, f"{len(comunes)} filas del test en el train de la app"
