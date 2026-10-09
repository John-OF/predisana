"""Revision 2026-10 — ningun calibrador puede afirmar 0% ni 100%, ni subir a saltos.

La isotonica a secas termina siempre en 0.0 y en 1.0: su primer escalon son los
scores mas bajos hasta el primer enfermo y el ultimo, los mas altos desde el ultimo
sano, tengan los puntos que tengan. En diabetes el 100% lo sostenia UNA persona del
train (un hombre de 68 con IMC 36 daba 45%; uno de 80 con IMC 50, 100%) y el 6% del
test de diabetes con glucosa recibia un 0,0% exacto.

`train_models.fit_calibrator` funde cada escalon extremo puro con su vecino y les da
la tasa real de los dos juntos. El resto de los escalones no se toca.

Despues une con rectas el centro de cada escalon con su nivel (isotonica centrada).
Con escalones el what-if subia a saltos: entre el 30% y el 56% de los pasos de un
barrido el modelo se movia y la probabilidad no, y en diabetes con glucosa un crudo de
0,868 daba 33% y uno de 0,870, 57%.
"""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

import modos as M
from train_models import fit_calibrator



def _a_secas(oof, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y).predict(oof)


# ---------- fit_calibrator ----------

def test_funde_cada_extremo_puro_con_su_vecino():
    """Escalones de la isotonica a secas: [0,0]=0 | [1,0]=0.5 | [1,1,0]=0.67 | [1,1]=1.
    Fundidos: los 4 primeros -> 1 de 4, centrado en 0,25; los 5 ultimos -> 4 de 5, en 0,7."""
    oof = np.arange(1, 10) / 10
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1])
    cal = fit_calibrator(oof, y)
    assert cal.X_thresholds_ == pytest.approx([0.25, 0.7])
    assert cal.y_thresholds_ == pytest.approx([0.25, 0.8])
    assert cal.predict([0.475]) == pytest.approx([0.525])   # entre los dos, una recta


def _datos_ruidosos(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    oof = rng.uniform(0.01, 0.99, n)
    return oof, (rng.uniform(size=n) < oof).astype(int)


def test_nunca_devuelve_cero_ni_uno():
    oof, y = _datos_ruidosos()
    cal = fit_calibrator(oof, y)
    # Tambien fuera del rango visto: out_of_bounds="clip" repite el nivel del borde.
    p = cal.predict(np.concatenate([oof, [0.0, 1.0]]))
    assert p.min() > 0.0
    assert p.max() < 1.0


def test_sigue_siendo_monotona():
    oof, y = _datos_ruidosos()
    p = fit_calibrator(oof, y).predict(np.sort(oof))
    assert (np.diff(p) >= 0).all()


def test_cada_escalon_conserva_su_nivel_en_su_centro():
    """Los escalones del interior son los de la isotonica tal cual (solo cambian los dos
    de cada extremo) y cada uno queda en la media de sus scores."""
    oof, y = _datos_ruidosos()
    a_secas = _a_secas(oof, y)
    niveles = np.unique(a_secas)
    assert len(niveles) > 6
    interior = niveles[2:-2]
    centros = [oof[a_secas == v].mean() for v in interior]
    cal = fit_calibrator(oof, y)
    assert cal.y_thresholds_[1:-1] == pytest.approx(interior, abs=1e-12)
    assert cal.X_thresholds_[1:-1] == pytest.approx(centros, abs=1e-12)
    assert cal.predict(centros) == pytest.approx(interior, abs=1e-12)


def test_el_escalon_fundido_es_la_tasa_real_del_grupo():
    oof, y = _datos_ruidosos()
    a_secas = _a_secas(oof, y)
    niveles = np.unique(a_secas)
    cal = fit_calibrator(oof, y)
    assert cal.y_thresholds_[0] == pytest.approx(y[a_secas <= niveles[1]].mean(), abs=1e-12)
    assert cal.y_thresholds_[-1] == pytest.approx(y[a_secas >= niveles[-2]].mean(), abs=1e-12)


def test_entre_el_primer_centro_y_el_ultimo_no_hay_mesetas():
    oof, y = _datos_ruidosos()
    cal = fit_calibrator(oof, y)
    x = np.linspace(cal.X_thresholds_[0], cal.X_thresholds_[-1], 2000)
    assert (np.diff(cal.predict(x)) > 0).all()


# ---------- los calibradores que se sirven ----------

def test_ningun_calibrador_servido_afirma_certeza(app_module):
    assert set(app_module.CALIBRATORS) == set(app_module.MODELS)
    for key, cal in app_module.CALIBRATORS.items():
        niveles = np.asarray(cal.y_thresholds_)
        assert niveles.min() > 0.0, f"{key} puede devolver 0%"
        assert niveles.max() < 1.0, f"{key} puede devolver 100%"


def test_ningun_calibrador_servido_tiene_mesetas(app_module):
    """Los de escalones guardaban los dos bordes de cada uno, con el mismo nivel."""
    for key, cal in app_module.CALIBRATORS.items():
        assert (np.diff(cal.X_thresholds_) > 0).all(), key
        assert (np.diff(cal.y_thresholds_) > 0).all(), key


def test_el_whatif_ya_no_sube_a_saltos(client, app_module):
    """Barrido de edad en cardiovascular completo (LogReg: el crudo sube en cada paso).
    Dentro del tramo entre el primer centro y el ultimo, la probabilidad tambien. Con
    escalones, en el test real entre el 30% y el 56% de los pasos salian planos."""
    curva = client.post("/whatif/cardiovascular?mode=completo", json={
        "base": COMPLETO, "feature": "age", "min": 20, "max": 80, "steps": 25}).get_json()["curve"]
    cal = app_module.CALIBRATORS["cardiovascular_completo"]
    dentro = [p for p in curva
              if cal.X_thresholds_[0] <= p["raw_probability"] <= cal.X_thresholds_[-1]]
    assert len(dentro) >= 20
    for antes, despues in zip(dentro, dentro[1:]):
        assert despues["raw_probability"] > antes["raw_probability"]
        assert despues["probability"] > antes["probability"]


_HOMBRE_FUMADOR = {"gender_Male": 1, "gender_Female": 0, "smoking_history_never": 0,
                   "smoking_history_current": 1, "smoking_history_former": 0}
_MUJER_NO_FUMADORA = {"gender_Male": 0, "gender_Female": 1, "smoking_history_never": 1,
                      "smoking_history_current": 0, "smoking_history_former": 0}
_TODOS_LOS_DX = {"diabetes": 1, "hypertension": 1, "high_cholesterol": 1, "heart_disease": 1}
_NINGUN_DX = {k: 0 for k in _TODOS_LOS_DX}

# Modo simplificado: peso y talla (200 kg y 160 cm dan un IMC de 78).
PEOR_CASO = {"age": 100, "weight": 200, "height": 160, **_TODOS_LOS_DX, **_HOMBRE_FUMADOR}
MEJOR_CASO = {"age": 18, "weight": 50, "height": 170, **_NINGUN_DX, **_MUJER_NO_FUMADORA}

COMPLETO = {"age": 40, "bmi": 30, "waist_circumference": 104, "ap_hi": 130, "ap_lo": 85,
            "total_cholesterol": 210, "hdl_cholesterol": 45, "hba1c_level": 5.6, "egfr": 90,
            "albumin_creatinine_ratio": 10, "alt": 25, "diabetes": 0, "hypertension": 0,
            "high_cholesterol": 1, "heart_disease": 0, **_HOMBRE_FUMADOR}
PEOR_COMPLETO = {"age": 90, "bmi": 60, "waist_circumference": 160, "ap_hi": 190, "ap_lo": 110,
                 "total_cholesterol": 320, "hdl_cholesterol": 25, "hba1c_level": 9, "egfr": 25,
                 "albumin_creatinine_ratio": 800, "alt": 300, **_TODOS_LOS_DX, **_HOMBRE_FUMADOR}
MEJOR_COMPLETO = {"age": 20, "bmi": 20, "waist_circumference": 70, "ap_hi": 105, "ap_lo": 65,
                  "total_cholesterol": 150, "hdl_cholesterol": 75, "hba1c_level": 5.0, "egfr": 120,
                  "albumin_creatinine_ratio": 4, "alt": 10, **_NINGUN_DX, **_MUJER_NO_FUMADORA}


@pytest.mark.parametrize("disease", list(M.ENFERMEDADES))
def test_ni_el_peor_perfil_da_100_ni_el_mejor_da_0(client, disease):
    peor = client.post(f"/predict/{disease}", json=PEOR_CASO).get_json()
    mejor = client.post(f"/predict/{disease}", json=MEJOR_CASO).get_json()
    assert 0.5 < peor["probability"] < 1.0
    assert 0.0 < mejor["probability"] < 0.1


@pytest.mark.parametrize("disease", list(M.ENFERMEDADES))
def test_el_modo_completo_tampoco_llega_a_los_extremos(client, disease):
    peor = client.post(f"/predict/{disease}?mode=completo", json=PEOR_COMPLETO).get_json()
    mejor = client.post(f"/predict/{disease}?mode=completo", json=MEJOR_COMPLETO).get_json()
    assert peor["mode"] == mejor["mode"] == "completo"
    assert 0.5 < peor["probability"] < 1.0
    assert 0.0 < mejor["probability"] < 0.1
