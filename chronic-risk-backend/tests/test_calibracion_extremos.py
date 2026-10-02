"""Revision 2026-10 — ningun calibrador puede afirmar 0% ni 100%.

La isotonica a secas termina siempre en 0.0 y en 1.0: su primer escalon son los
scores mas bajos hasta el primer enfermo y el ultimo, los mas altos desde el ultimo
sano, tengan los puntos que tengan. En diabetes el 100% lo sostenia UNA persona del
train (un hombre de 68 con IMC 36 daba 45%; uno de 80 con IMC 50, 100%) y el 6% del
test de diabetes con glucosa recibia un 0,0% exacto.

`train_models.fit_calibrator` funde cada escalon extremo puro con su vecino y les da
la tasa real de los dos juntos. El resto de la curva no se toca.
"""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from train_models import fit_calibrator


# ---------- fit_calibrator ----------

def test_funde_cada_extremo_puro_con_su_vecino():
    """Escalones de la isotonica a secas: [0,0]=0 | [1,0]=0.5 | [1,1,0]=0.67 | [1,1]=1.
    Fundidos: los 4 primeros -> 1 de 4; los 5 ultimos -> 4 de 5."""
    oof = np.arange(1, 10) / 10
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1])
    cal = fit_calibrator(oof, y)
    assert cal.predict(oof) == pytest.approx([0.25] * 4 + [0.8] * 5)


def _datos_ruidosos(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    oof = rng.uniform(0.01, 0.99, n)
    return oof, (rng.uniform(size=n) < oof).astype(int)


def test_nunca_devuelve_cero_ni_uno():
    oof, y = _datos_ruidosos()
    cal = fit_calibrator(oof, y)
    # Tambien fuera del rango visto: out_of_bounds="clip" repite el escalon del borde.
    p = cal.predict(np.concatenate([oof, [0.0, 1.0]]))
    assert p.min() > 0.0
    assert p.max() < 1.0


def test_sigue_siendo_monotona():
    oof, y = _datos_ruidosos()
    p = fit_calibrator(oof, y).predict(np.sort(oof))
    assert (np.diff(p) >= 0).all()


def test_el_interior_de_la_curva_no_se_toca():
    """Solo cambian los dos escalones de cada extremo; el resto es la isotonica tal cual."""
    oof, y = _datos_ruidosos()
    a_secas = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y).predict(oof)
    niveles = np.unique(a_secas)
    assert len(niveles) > 6
    interior = (a_secas > niveles[1]) & (a_secas < niveles[-2])
    nuevo = fit_calibrator(oof, y).predict(oof)
    assert interior.sum() > 0.8 * len(oof)
    assert nuevo[interior] == pytest.approx(a_secas[interior], abs=1e-12)
    assert (nuevo[~interior] != a_secas[~interior]).any()


def test_el_escalon_fundido_es_la_tasa_real_del_grupo():
    oof, y = _datos_ruidosos()
    nuevo = fit_calibrator(oof, y).predict(oof)
    for extremo in (nuevo.min(), nuevo.max()):
        grupo = nuevo == extremo
        assert extremo == pytest.approx(y[grupo].mean(), abs=1e-12)


# ---------- los calibradores que se sirven ----------

def test_ningun_calibrador_servido_afirma_certeza(app_module):
    assert set(app_module.CALIBRATORS) == set(app_module.MODELS)
    for key, cal in app_module.CALIBRATORS.items():
        niveles = np.asarray(cal.y_thresholds_)
        assert niveles.min() > 0.0, f"{key} puede devolver 0%"
        assert niveles.max() < 1.0, f"{key} puede devolver 100%"


PEOR_CASO = {
    "diabetes": {"age": 100, "bmi": 90, "hypertension": 1, "heart_disease": 1,
                 "gender_Male": 1, "gender_Female": 0, "smoking_history_never": 0,
                 "smoking_history_current": 1, "smoking_history_former": 0},
    "hipertension": {"age": 100, "bmi": 90, "weight": 250, "waist_circumference": 200,
                     "diabetes": 1, "heart_disease": 1, "high_cholesterol": 1,
                     "gender_Male": 1, "gender_Female": 0, "smoking_history_never": 0,
                     "smoking_history_current": 1, "smoking_history_former": 0},
    "cardiovascular": {"age": 100, "bmi": 90, "ap_hi": 250, "ap_lo": 150,
                       "cholesterol": 3, "gluc": 3, "smoke": 1, "alco": 1, "active": 0,
                       "gender_Male": 1, "gender_Female": 0},
}
MEJOR_CASO = {
    "diabetes": {"age": 18, "bmi": 18, "hypertension": 0, "heart_disease": 0,
                 "gender_Male": 0, "gender_Female": 1, "smoking_history_never": 1,
                 "smoking_history_current": 0, "smoking_history_former": 0},
    "hipertension": {"age": 18, "bmi": 18, "weight": 50, "waist_circumference": 62,
                     "diabetes": 0, "heart_disease": 0, "high_cholesterol": 0,
                     "gender_Male": 0, "gender_Female": 1, "smoking_history_never": 1,
                     "smoking_history_current": 0, "smoking_history_former": 0},
    "cardiovascular": {"age": 30, "bmi": 19, "ap_hi": 95, "ap_lo": 60,
                       "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
                       "gender_Male": 0, "gender_Female": 1},
}


@pytest.mark.parametrize("disease", ["diabetes", "hipertension", "cardiovascular"])
def test_ni_el_peor_perfil_da_100_ni_el_mejor_da_0(client, disease):
    peor = client.post(f"/predict/{disease}", json=PEOR_CASO[disease]).get_json()
    mejor = client.post(f"/predict/{disease}", json=MEJOR_CASO[disease]).get_json()
    assert 0.5 < peor["probability"] < 1.0
    assert 0.0 < mejor["probability"] < 0.1


def test_diabetes_con_glucosa_tampoco_llega_a_los_extremos(client):
    alta = client.post("/predict/diabetes", json={**PEOR_CASO["diabetes"], "blood_glucose_level": 500}).get_json()
    baja = client.post("/predict/diabetes", json={**MEJOR_CASO["diabetes"], "blood_glucose_level": 75}).get_json()
    assert alta["variant"] == baja["variant"] == "glucosa"
    assert 0.5 < alta["probability"] < 1.0
    assert 0.0 < baja["probability"] < 0.1
