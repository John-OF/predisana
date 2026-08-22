# Regresiones de la auditoria 2026-08-20 (AUD-1, 2, 3, 5, 6, 7, 8, 9, 14, 15 y 18).
# Cada test fija el comportamiento CORREGIDO para que no vuelva a colarse.
import json
import math

import pytest


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


# ---------- AUD-3: entrada invalida -> 400, enfermedad inexistente -> 404 ----------

@pytest.mark.parametrize("valor", ["cuarenta", [1, 2, 3], {"a": 1}, "NaN", float("inf")])
def test_predict_valor_no_numerico_da_400(client, perfil_diabetes, valor):
    r = client.post("/predict/diabetes", json={**perfil_diabetes, "age": valor})
    assert r.status_code == 400
    assert "age" in r.get_json()["error"]

def test_predict_campo_vacio_se_trata_como_ausente(client, perfil_diabetes):
    """Un input borrado en el form manda "": es ausente, no un 500."""
    r = client.post("/predict/diabetes", json={**perfil_diabetes, "bmi": ""})
    assert r.status_code == 200
    assert "bmi" in r.get_json()["missing_filled_as_zero"]

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
                                              "base": {**perfil_diabetes, "bmi": "gordo"}})
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
    assert "diabetes" in d["models_loaded"]


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
    assert guardado["age"] == 55 and guardado["bmi"] == 31

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

PERFIL_HTA = {
    "age": 50, "bmi": 28, "weight": 80, "waist_circumference": 95,
    "diabetes": 0, "heart_disease": 0, "high_cholesterol": 0,
    "gender_Male": 1, "gender_Female": 0,
    "smoking_history_never": 1, "smoking_history_current": 0, "smoking_history_former": 0,
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

def test_hipertension_el_riesgo_crece_con_el_imc(client):
    riesgos = [_riesgo_hta(client, bmi=b) for b in (20, 27, 33, 40)]
    assert riesgos == sorted(riesgos), riesgos

def test_hipertension_no_satura_en_el_extremo_sano(client):
    """Un adulto joven y delgado no puede salir con un riesgo alto (el modelo viejo
    devolvia 100% con 25 años y presion 110)."""
    assert _riesgo_hta(client, age=25, bmi=22, weight=62, waist_circumference=75) < 0.15

def test_hipertension_las_comorbilidades_suman(client):
    base = _riesgo_hta(client)
    assert _riesgo_hta(client, diabetes=1) > base
    assert _riesgo_hta(client, high_cholesterol=1) > base

def test_presion_no_entra_al_modelo_pero_si_a_la_capa_clinica(client):
    sin = client.post("/predict/hipertension", json=PERFIL_HTA).get_json()
    con = client.post("/predict/hipertension",
                      json={**PERFIL_HTA, "blood_pressure": 165}).get_json()
    assert con["probability"] == sin["probability"]      # la presion no mueve el modelo
    assert not sin["clinical_flags"]
    assert any(f["indicator"] == "blood_pressure" for f in con["clinical_flags"])
    assert con["clinical_flags"][0]["source"] == "ACC/AHA"


# ---------- AUD-14: tope fisiologico de la glucosa en los datos REALES ----------
@pytest.mark.parametrize("ruta", [
    "data_processed/diabetes_dataset.csv",
    "data_curated/diabetes/diabetes_train.csv",
    "data_curated/diabetes/diabetes_test.csv",
])
def test_glucosa_real_topada(ruta):
    """NHANES trae hasta 898 mg/dL. El tope ya se aplicaba al sintetico y a la entrada
    del formulario, pero no al CSV real, y esas filas acababan en el fondo de SHAP y en
    las fichas de 'caso real' del laboratorio."""
    import pandas as pd
    g = pd.read_csv(ruta)["blood_glucose_level"]
    assert g.max() <= 500, f"{ruta}: glucosa maxima {g.max()}"


def test_el_fondo_de_shap_no_tiene_glucosas_imposibles(app_module):
    """El fondo sale del train curado: si entra un 898, distorsiona las explicaciones."""
    import numpy as np
    feats = app_module.FEATURES["diabetes_glucosa"]
    fondo = app_module._load_background_for_shap("diabetes", feats)
    if "blood_glucose_level" in feats:
        col = fondo[:, feats.index("blood_glucose_level")]
        assert np.nanmax(col) <= 500


# ---------- AUD-15: un solo split para diabetes ----------
# `train_nhanes_diabetes.py` hacia su propio train_test_split del CSV completo
# mientras `data_curated/diabetes/*` salia de `curate_and_synthesize.py`. Para la
# variante con glucosa los dos repartos eran distintos (filtraba las filas sin
# glucosa ANTES de partir, asi que n cambiaba): el test que reportaban las metricas
# no era el curado, y el 79% de sus filas de test estaban en el fondo de SHAP.
FEATURES_VARIANTE = {
    "diabetes": ["age", "bmi", "hypertension", "heart_disease",
                 "gender_Female", "gender_Male", "smoking_history_never",
                 "smoking_history_current", "smoking_history_former"],
    "diabetes_glucosa": ["age", "bmi", "hypertension", "heart_disease",
                         "gender_Female", "gender_Male", "smoking_history_never",
                         "smoking_history_current", "smoking_history_former",
                         "blood_glucose_level"],
}


def _curado(cual):
    import pandas as pd
    return pd.read_csv(f"data_curated/diabetes/diabetes_{cual}.csv")


def test_train_y_test_curados_son_disjuntos():
    import pandas as pd
    tr, te = _curado("train"), _curado("test")
    cols = list(tr.columns)
    comunes = pd.merge(tr[cols].round(6), te[cols].round(6), how="inner")
    assert comunes.empty, f"{len(comunes)} filas compartidas entre train y test"


@pytest.mark.parametrize("clave", ["diabetes", "diabetes_glucosa"])
def test_las_metricas_se_reportan_sobre_el_test_curado(clave):
    """El numero publicado en /metricas tiene que salir del MISMO test que sirve el
    laboratorio. Antes, `diabetes_glucosa` reportaba sobre 1122 filas de otro reparto
    mientras el test curado tenia 1131."""
    feats = FEATURES_VARIANTE[clave]
    esperado = len(_curado("test").dropna(subset=feats + ["target"]))
    with open(f"models/{clave}_metrics.json", encoding="utf-8") as f:
        reportado = json.load(f)["report_test"]["macro avg"]["support"]
    assert int(reportado) == esperado


def test_el_fondo_de_shap_no_contiene_filas_del_test(app_module):
    """El fondo sale del train curado; si ese train no es el del modelo, las
    explicaciones se calculan contra filas que el modelo uso para evaluarse."""
    import numpy as np
    import pandas as pd
    feats = app_module.FEATURES["diabetes_glucosa"]
    fondo = pd.DataFrame(app_module._load_background_for_shap("diabetes", feats),
                         columns=feats)
    test = _curado("test").dropna(subset=feats + ["target"])[feats]
    comunes = pd.merge(fondo.round(6), test.round(6), how="inner")
    assert comunes.empty, f"{len(comunes)} filas del test en el fondo de SHAP"
