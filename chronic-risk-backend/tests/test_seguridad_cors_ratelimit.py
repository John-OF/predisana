"""AUD-4 — CORS restringido en /admin/* y rate limiting por IP.

Antes: `CORS(app)` abria TODO con "*" (incluido el admin) y no habia ningun limite,
asi que el ADMIN_TOKEN se podia fuerza-brutear desde cualquier origen sin freno.

Cada test usa su propia IP falsa (`environ_base={"REMOTE_ADDR": ...}`) para no
compartir cubeta con el resto de la suite ni con los demas tests de este archivo.
"""
import pytest

EVIL = "https://sitio-hostil.example"
DEV = "http://localhost:5173"


def _acao(resp):
    """Access-Control-Allow-Origin de la respuesta (None si no la hay)."""
    return resp.headers.get("Access-Control-Allow-Origin")


# ---------- CORS ----------
def test_endpoints_publicos_siguen_abiertos(client):
    """El API publico es una demo: sin CORS_ORIGINS configurado, sigue abierto."""
    r = client.get("/config/diabetes", headers={"Origin": EVIL})
    assert r.status_code == 200
    assert _acao(r) is not None


def test_admin_no_hereda_el_asterisco(client, admin_headers):
    """Un origen cualquiera NO recibe cabecera CORS en /admin/*: el navegador le
    impide leer la respuesta aunque adivinara el token."""
    r = client.get(
        "/admin/stats",
        headers={**admin_headers, "Origin": EVIL},
        environ_base={"REMOTE_ADDR": "198.51.100.10"},
    )
    assert _acao(r) is None


def test_admin_permite_el_origen_de_desarrollo(client, admin_headers):
    r = client.get(
        "/admin/stats",
        headers={**admin_headers, "Origin": DEV},
        environ_base={"REMOTE_ADDR": "198.51.100.11"},
    )
    assert r.status_code == 200
    assert _acao(r) == DEV


def test_preflight_admin_rechaza_origen_desconocido(client):
    """El preflight del navegador tampoco autoriza el X-Admin-Token para un origen
    que no esta en la lista."""
    cab = {"Access-Control-Request-Method": "GET",
           "Access-Control-Request-Headers": "x-admin-token"}
    r_evil = client.options("/admin/verify", headers={**cab, "Origin": EVIL})
    r_dev = client.options("/admin/verify", headers={**cab, "Origin": DEV})
    assert _acao(r_evil) is None
    assert _acao(r_dev) == DEV


def test_admin_rechaza_origen_no_permitido_en_servidor(client, admin_headers):
    """Defensa en profundidad: CORS solo oculta la respuesta, la peticion se ejecuta
    igual. Con Origin hostil el servidor corta con 403 antes de mirar el token."""
    r = client.get(
        "/admin/stats",
        headers={**admin_headers, "Origin": EVIL},
        environ_base={"REMOTE_ADDR": "198.51.100.12"},
    )
    assert r.status_code == 403
    assert "origen" in r.get_json()["error"].lower()


def test_sin_cabecera_origin_el_admin_funciona(client, admin_headers):
    """curl / el propio deploy no mandan Origin: ahi manda el token, no el CORS."""
    r = client.get(
        "/admin/stats", headers=admin_headers,
        environ_base={"REMOTE_ADDR": "198.51.100.13"},
    )
    assert r.status_code == 200


def test_origenes_desde_env(app_module, monkeypatch):
    """El parser de la env var: lista separada por comas, ignora espacios y vacios."""
    monkeypatch.setenv("X_ORIGENES", " https://a.example , https://b.example ,, ")
    assert app_module._origenes_desde_env("X_ORIGENES", ["defecto"]) == [
        "https://a.example", "https://b.example"]
    monkeypatch.setenv("X_ORIGENES", "   ")
    assert app_module._origenes_desde_env("X_ORIGENES", ["defecto"]) == ["defecto"]
    monkeypatch.delenv("X_ORIGENES")
    assert app_module._origenes_desde_env("X_ORIGENES", ["defecto"]) == ["defecto"]


def test_admin_nunca_abre_con_asterisco(app_module):
    """Aunque alguien ponga CORS_ORIGINS=* , el admin no debe quedar en '*'."""
    assert "*" not in app_module.ADMIN_CORS_ORIGINS


# ---------- Rate limiting ----------
def test_fuerza_bruta_del_token_se_corta(client, app_module):
    """El caso de AUD-4: probar tokens hasta acertar. A partir del limite -> 429."""
    limite = int(app_module.RATE_LIMIT_ADMIN_VERIFY.split()[0])
    ip = {"REMOTE_ADDR": "203.0.113.7"}
    codigos = [
        client.get("/admin/verify", headers={"X-Admin-Token": f"intento-{i}"},
                   environ_base=ip).status_code
        for i in range(limite + 2)
    ]
    assert codigos[:limite] == [401] * limite       # los primeros, no autorizado
    assert codigos[limite:] == [429, 429]           # despues, cortado


def test_respuesta_429_es_json(client, app_module):
    limite = int(app_module.RATE_LIMIT_ADMIN_VERIFY.split()[0])
    ip = {"REMOTE_ADDR": "203.0.113.8"}
    for _ in range(limite):
        client.get("/admin/verify", headers={"X-Admin-Token": "x"}, environ_base=ip)
    r = client.get("/admin/verify", headers={"X-Admin-Token": "x"}, environ_base=ip)
    assert r.status_code == 429
    assert r.is_json and "error" in r.get_json()
    assert r.headers.get("Retry-After")             # el cliente sabe cuanto esperar


def test_el_limite_es_por_ip(client, admin_headers, app_module):
    """Agotar la cubeta de una IP no debe afectar a otra (si no, un solo abusador
    dejaria fuera a todos los visitantes de la demo)."""
    limite = int(app_module.RATE_LIMIT_ADMIN_VERIFY.split()[0])
    quemada = {"REMOTE_ADDR": "203.0.113.9"}
    for _ in range(limite + 1):
        client.get("/admin/verify", headers={"X-Admin-Token": "x"}, environ_base=quemada)
    assert client.get("/admin/verify", headers={"X-Admin-Token": "x"},
                      environ_base=quemada).status_code == 429
    otra = client.get("/admin/verify", headers=admin_headers,
                      environ_base={"REMOTE_ADDR": "203.0.113.10"})
    assert otra.status_code == 200


def test_health_exento_del_limite(client, app_module):
    """Los monitores de uptime pinchan /health en bucle: un 429 ahi marcaria el
    deploy como caido."""
    assert app_module.limiter.limit is not None
    ip = {"REMOTE_ADDR": "203.0.113.11"}
    codigos = {client.get("/health", environ_base=ip).status_code for _ in range(40)}
    assert codigos == {200}


def test_predict_tiene_su_propio_limite(client, perfil_diabetes, app_module):
    """/predict escribe una fila en la BD por request anonimo: debe estar limitado
    (AUD-9 dejo el tope de tamano; el de frecuencia es este)."""
    esperado = app_module.RATE_LIMIT_PREDICT.split()[0]
    ip = {"REMOTE_ADDR": "203.0.113.12"}
    r = client.post("/predict/diabetes", json=perfil_diabetes, environ_base=ip)
    assert r.status_code == 200
    # La cabecera delata que la cubeta que se aplico es la de /predict, no la global.
    assert r.headers.get("X-RateLimit-Limit") == esperado
    r = client.post("/whatif/diabetes",
                    json={"base": perfil_diabetes, "feature": "age",
                          "min": 30, "max": 70, "steps": 5},
                    environ_base=ip)
    assert r.status_code == 200
    assert r.headers.get("X-RateLimit-Limit") == esperado
    # /health, exento, no lleva cabeceras de limite.
    assert client.get("/health", environ_base=ip).headers.get("X-RateLimit-Limit") is None
