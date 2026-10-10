# Tests unitarios de los helpers puros de app.py (sin HTTP).
import pytest


# ---------- claves de modelo: enfermedad y modo ----------

def test_clave_y_partes_son_inversas(app_module):
    for disease in app_module.ENFERMEDADES:
        for modo in app_module.MODOS:
            assert app_module._partes(app_module._clave(disease, modo)) == (disease, modo)

def test_hay_un_modelo_por_enfermedad_y_modo(app_module):
    assert sorted(app_module.CLAVES) == sorted(app_module.MODELS)
    assert len(app_module.CLAVES) == len(app_module.ENFERMEDADES) * len(app_module.MODOS)


# ---------- el IMC del simplificado sale del peso y la talla ----------

def test_derivar_calcula_el_imc(app_module):
    datos, dado = app_module._derivar("simplificado", {"weight": 64, "height": 160})
    assert datos["bmi"] == 25.0 and dado is False

def test_derivar_respeta_un_imc_dado(app_module):
    datos, dado = app_module._derivar("simplificado", {"weight": 64, "height": 160, "bmi": 30})
    assert datos["bmi"] == 30 and dado is True

def test_derivar_no_toca_el_completo(app_module):
    """El completo pide el IMC medido: el peso y la talla no lo calculan."""
    datos, _ = app_module._derivar("completo", {"weight": 64, "height": 160})
    assert "bmi" not in datos

def test_derivar_sin_talla_no_inventa_nada(app_module):
    datos, _ = app_module._derivar("simplificado", {"weight": 64})
    assert "bmi" not in datos

def test_derivar_rechaza_un_imc_imposible(app_module):
    with pytest.raises(app_module.InvalidPayload, match="IMC"):
        app_module._derivar("simplificado", {"weight": 250, "height": 120})


# ---------- capa clinica ADA / ACC-AHA (separada del modelo) ----------

@pytest.mark.parametrize("glucosa,categoria", [
    (250, "diabetes"),        # >=200: compatible con diabetes
    (150, "diabetes"),        # >=126: criterio de diabetes
    (110, "prediabetes"),     # 100-125
    (90, None),               # normal: sin flag
])
def test_flags_glucosa_ada(app_module, glucosa, categoria):
    flags = app_module.compute_clinical_flags(glucosa, 0, 0)
    cats = [f["category"] for f in flags if f["indicator"] == "glucose"]
    assert cats == ([categoria] if categoria else [])

@pytest.mark.parametrize("hba1c,categoria", [
    (7.0, "diabetes"),
    (6.0, "prediabetes"),
    (5.0, None),
])
def test_flags_hba1c_ada(app_module, hba1c, categoria):
    flags = app_module.compute_clinical_flags(0, hba1c, 0)
    cats = [f["category"] for f in flags if f["indicator"] == "hba1c"]
    assert cats == ([categoria] if categoria else [])

@pytest.mark.parametrize("sistolica,categoria", [
    (185, "crisis_hipertensiva"),
    (150, "hipertension_estadio_2"),
    (135, "hipertension_estadio_1"),
    (125, "presion_elevada"),
    (110, None),
])
def test_flags_presion_acc_aha(app_module, sistolica, categoria):
    flags = app_module.compute_clinical_flags(0, 0, sistolica)
    cats = [f["category"] for f in flags if f["indicator"] == "blood_pressure"]
    assert cats == ([categoria] if categoria else [])

def test_flags_citan_fuente(app_module):
    flags = app_module.compute_clinical_flags(250, 7.0, 185)
    assert {f["source"] for f in flags} == {"ADA", "ACC/AHA"}


# ---------- _safe_get (lookup tolerante de features en el payload) ----------

def test_safe_get_exacto(app_module):
    assert app_module._safe_get({"age": 40}, "age") == 40

def test_safe_get_case_insensitive(app_module):
    assert app_module._safe_get({"Age": 40}, "age") == 40

def test_safe_get_ausente(app_module):
    assert app_module._safe_get({"age": 40}, "bmi") is None


# ---------- presion: manda la mas alta de las dos (v2) ----------

@pytest.mark.parametrize("sistolica,diastolica,categoria", [
    (132, 95, "hipertension_estadio_2"),   # el objetivo de hipertension es >= 140/90
    (118, 85, "hipertension_estadio_1"),
    (0, 92, "hipertension_estadio_2"),     # solo la diastolica
    (125, 70, "presion_elevada"),
    (115, 75, None),
])
def test_flags_presion_con_la_diastolica(app_module, sistolica, diastolica, categoria):
    flags = app_module.compute_clinical_flags(0, 0, sistolica, diastolica)
    cats = [f["category"] for f in flags if f["indicator"] == "blood_pressure"]
    assert cats == ([categoria] if categoria else [])


# ---------- funcion renal: KDIGO (enfermedad renal cronica) ----------

@pytest.mark.parametrize("egfr,categoria", [
    (10, "filtrado_G5"),
    (22, "filtrado_G4"),
    (38, "filtrado_G3b"),
    (59.9, "filtrado_G3a"),
    (60, None),          # 60-89 (G2) solo es enfermedad con albuminuria
    (95, None),
    (0, None),           # 0 = no lo aporto
])
def test_flags_filtrado_kdigo(app_module, egfr, categoria):
    flags = app_module.compute_clinical_flags(0, 0, 0, egfr=egfr)
    cats = [f["category"] for f in flags if f["indicator"] == "egfr"]
    assert cats == ([categoria] if categoria else [])


@pytest.mark.parametrize("acr,categoria", [
    (450, "albuminuria_A3"),
    (300, "albuminuria_A2"),
    (30, "albuminuria_A2"),
    (29.9, None),
])
def test_flags_albuminuria_kdigo(app_module, acr, categoria):
    flags = app_module.compute_clinical_flags(0, 0, 0, albumin_creatinine=acr)
    cats = [f["category"] for f in flags if f["indicator"] == "albuminuria"]
    assert cats == ([categoria] if categoria else [])


def test_kdigo_recuerda_que_una_medicion_no_basta(app_module):
    """El diagnostico exige que se mantenga 3 meses: un solo analisis no lo es."""
    for flag in app_module.compute_clinical_flags(0, 0, 0, egfr=45, albumin_creatinine=80):
        assert flag["source"] == "KDIGO" and "3 meses" in flag["detail"]


def test_enumerar_en_castellano(app_module):
    assert app_module._enumerar(["a"]) == "a"
    assert app_module._enumerar(["a", "b", "c"]) == "a, b y c"
