# Regresiones de la auditoria 2026-08-20 (AUD-2, AUD-3, AUD-5, AUD-6, AUD-8, AUD-18).
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
