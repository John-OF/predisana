# Tests del panel admin (A3): auth por token, logueo de predicciones con
# session_id, stats agregadas y export CSV. Corre contra la BD temporal.


# ---------- auth ----------

def test_admin_sin_token_401(client):
    assert client.get("/admin/verify").status_code == 401

def test_admin_token_malo_401(client):
    r = client.get("/admin/verify", headers={"X-Admin-Token": "incorrecto"})
    assert r.status_code == 401

def test_admin_token_correcto_200(client, admin_headers):
    assert client.get("/admin/verify", headers=admin_headers).status_code == 200

def test_admin_deshabilitado_503(client, app_module, monkeypatch):
    """Sin ADMIN_TOKEN configurado, el panel entero queda deshabilitado."""
    monkeypatch.setattr(app_module, "ADMIN_TOKEN", None)
    r = client.get("/admin/verify", headers={"X-Admin-Token": "da igual"})
    assert r.status_code == 503


# ---------- flujo: /predict loguea -> el admin lo ve ----------

def test_predict_se_loguea_y_admin_lo_lista(client, admin_headers, perfil_diabetes):
    sesion = "sesion-pytest-123"
    r = client.post("/predict/diabetes",
                    json={**perfil_diabetes, "blood_glucose_level": 170},
                    headers={"X-Session-Id": sesion})
    assert r.status_code == 200

    r = client.get("/admin/predictions?limit=5&disease=diabetes", headers=admin_headers)
    assert r.status_code == 200
    items = r.get_json()["items"]
    assert items, "el admin no devolvio simulaciones"
    ultimo = items[0]  # orden: mas reciente primero
    assert ultimo["disease"] == "diabetes"
    assert ultimo["session_id"] == sesion
    assert ultimo["model"], "la fila no guardo el modelo servido"
    assert ultimo["clinical_note"], "la fila no guardo la nota clinica"
    assert isinstance(ultimo["input_data"], dict)


def test_admin_stats_agrega_coherente(client, admin_headers):
    r = client.get("/admin/stats", headers=admin_headers)
    assert r.status_code == 200
    s = r.get_json()
    assert s["total"] >= 1
    assert s["distinct_sessions"] >= 1
    por_enfermedad = {b["disease"]: b for b in s["by_disease"]}
    # Solo enfermedades servidas (nunca 'obesidad' u otras retiradas).
    assert set(por_enfermedad) == {"diabetes", "hipertension", "cardiovascular"}
    # El histograma de cada enfermedad debe sumar exactamente su conteo.
    for d, fila in por_enfermedad.items():
        assert sum(s["prob_histogram"][d]) == fila["count"]
    assert sum(s["hourly"]) == s["total"]


def test_admin_export_csv(client, admin_headers):
    r = client.get("/admin/export.csv", headers=admin_headers)
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    lineas = r.get_data(as_text=True).strip().splitlines()
    assert lineas[0].startswith("id,timestamp,disease")
    assert len(lineas) >= 2  # cabecera + al menos una simulacion logueada
