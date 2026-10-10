# Tests del panel admin (A3): auth por token, logueo de predicciones con
# session_id, stats agregadas y export CSV. Corre contra la BD temporal.
import modos as M


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


def test_paciente_sintetico_del_laboratorio_no_se_registra(client, app_module, perfil_diabetes):
    """Auditoria 2026-09: el laboratorio (Proyecto.jsx) evaluaba pacientes del GAN con
    /predict y quedaban guardados como simulaciones de usuarios, mezclados en la
    analitica del admin. Con ?source=synthetic se evaluan igual pero no se guardan."""
    def filas():
        with app_module.SessionLocal() as s:
            return s.query(app_module.Prediction).count()

    antes = filas()
    sintetico = client.post("/predict/diabetes?source=synthetic", json=perfil_diabetes)
    assert sintetico.status_code == 200
    assert filas() == antes
    usuario = client.post("/predict/diabetes", json=perfil_diabetes)
    assert filas() == antes + 1
    assert sintetico.get_json()["probability"] == usuario.get_json()["probability"]


def test_admin_stats_agrega_coherente(client, admin_headers):
    r = client.get("/admin/stats", headers=admin_headers)
    assert r.status_code == 200
    s = r.get_json()
    assert s["total"] >= 1
    assert s["distinct_sessions"] >= 1
    por_enfermedad = {b["disease"]: b for b in s["by_disease"]}
    # Solo enfermedades servidas (nunca 'obesidad' u otras retiradas).
    assert set(por_enfermedad) == set(M.ENFERMEDADES)
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


# ---------- lo que ve el admin cuadra con lo que vio el usuario (revision 2026-10) ----------

def test_admin_lista_la_banda_que_vio_el_usuario_y_la_hora_en_utc(client, admin_headers, perfil_diabetes):
    """La columna "Resultado" era la clase cruda del modelo (>= 0,5 sin calibrar), que el
    simulador no muestra; y la hora, UTC sin la Z, se leia como local."""
    vista = client.post("/predict/diabetes", json=perfil_diabetes).get_json()
    [fila] = client.get("/admin/predictions?limit=1", headers=admin_headers).get_json()["items"]
    assert fila["risk_band"] == vista["risk_band"]
    assert fila["timestamp"].endswith("Z")


def test_admin_stats_reparte_por_banda_y_por_modo(client, admin_headers, perfil_diabetes):
    client.post("/predict/diabetes?mode=simplificado", json=perfil_diabetes)
    s = client.get("/admin/stats", headers=admin_headers).get_json()
    assert s["timezone"] == "UTC"
    for fila in s["by_disease"]:
        assert sum(fila["bands"].values()) == fila["count"]
        assert sum(fila["modes"].values()) <= fila["count"]   # filas viejas sin modo
        assert set(fila["risk_bands"]) >= {"low_below", "high_from"}
    diabetes = next(f for f in s["by_disease"] if f["disease"] == "diabetes")
    assert diabetes["modes"]["simplificado"] >= 1


def test_el_top_de_factores_solo_cuenta_variables_de_los_modelos_servidos(client, app_module, admin_headers):
    """Una fila de la v1 (con la glucosa como variable) no puede colarse en el top."""
    import json
    with app_module.SessionLocal() as s:
        s.add(app_module.Prediction(disease="diabetes", probability=0.4, prediction=0,
                                    top_features=json.dumps([{"feature": "blood_glucose_level", "shap": 2.0},
                                                             {"feature": "cholesterol", "shap": 1.0}])))
        s.commit()
    top = {f["feature"] for f in client.get("/admin/stats", headers=admin_headers).get_json()["top_features"]}
    assert not {"blood_glucose_level", "cholesterol"} & top

