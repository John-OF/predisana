# coherence.py
# Combinaciones de datos que no pueden ser de una misma persona, aunque cada dato por
# separado caiga en su rango. Vive en su propio modulo porque lo usan dos sitios que
# no pueden importarse entre si:
#   - app.py: el aviso `incoherente` de /predict.
#   - curate_and_synthesize.py: descarta del pool del GAN las filas que el propio
#     simulador marcaria. Un "caso virtual" no puede llegar con una talla de 0,94 m.
#
# Las reglas miran los campos que haya, no la enfermedad: cada modo pide variables
# distintas (el simplificado, peso y talla; el completo, ademas cintura y presion).
#   - Peso e IMC juntos implican una talla: si el IMC no sale de ese peso y esa talla
#     (porque se mando el IMC a mano), tiene que dar una talla humana.
#   - Cintura e IMC: la v1 pedia peso, IMC y cintura sueltos, y el modelo de
#     hipertension aprendio con ellos un coeficiente NEGATIVO para el peso.
#   - Presion: la diastolica no puede igualar ni superar a la sistolica (auditoria
#     2026-08, corregido en la revision 2026-10).
import math
from typing import Any, Dict, List, Optional

ALTURA_IMPLICITA_RANGO_M = (1.30, 2.20)  # weight/bmi implican una altura fuera de esto -> incoherente
CINTURA_IMC_BAJO_MAX = 22.0
CINTURA_IMC_BAJO_MIN_CINTURA = 100.0
CINTURA_IMC_ALTO_MIN = 35.0
CINTURA_IMC_ALTO_MAX_CINTURA = 80.0

# Campos que miran las reglas. Los que falten no se juzgan.
CAMPOS = ("weight", "bmi", "waist_circumference", "ap_hi", "ap_lo")


def _corporales(weight, bmi, waist) -> List[Dict[str, Any]]:
    avisos: List[Dict[str, Any]] = []

    if weight is not None and bmi is not None:
        altura_implicita = math.sqrt(weight / bmi)
        lo, hi = ALTURA_IMPLICITA_RANGO_M
        if not (lo <= altura_implicita <= hi):
            avisos.append({
                "feature": "weight", "value": weight,
                "detail": (f"El peso ({weight:g} kg) y el IMC ({bmi:g}) juntos implican una "
                           f"altura de ~{altura_implicita:.2f} m, fuera de un rango humano "
                           f"plausible: revisa el peso, la talla o el IMC."),
            })

    if bmi is not None and waist is not None:
        if bmi <= CINTURA_IMC_BAJO_MAX and waist >= CINTURA_IMC_BAJO_MIN_CINTURA:
            avisos.append({
                "feature": "waist_circumference", "value": waist,
                "detail": (f"Cintura de {waist:g} cm con un IMC de {bmi:g} es una combinación "
                           f"casi imposible fisiológicamente: un IMC tan bajo no deja margen "
                           f"para tanta grasa abdominal."),
            })
        elif bmi >= CINTURA_IMC_ALTO_MIN and waist <= CINTURA_IMC_ALTO_MAX_CINTURA:
            avisos.append({
                "feature": "waist_circumference", "value": waist,
                "detail": (f"Cintura de {waist:g} cm con un IMC de {bmi:g} es una combinación "
                           f"casi imposible fisiológicamente: un IMC tan alto casi siempre "
                           f"viene con más cintura."),
            })

    return avisos


def _de_presion(ap_hi, ap_lo) -> List[Dict[str, Any]]:
    if ap_hi is None or ap_lo is None or ap_hi > ap_lo:
        return []
    return [{
        "feature": "ap_lo", "value": ap_lo,
        "detail": (f"La presión diastólica ({ap_lo:g}) no puede ser igual o mayor que la "
                   f"sistólica ({ap_hi:g}): revisa si están intercambiadas."),
    }]


def incoherencias(valores: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    """Una entrada {feature, value, detail} por cada combinacion incoherente.
    `valores` trae los campos de CAMPOS ya numericos y positivos, o None si faltan:
    sin los dos datos de una combinacion, esa combinacion no se juzga."""
    return (_corporales(valores.get("weight"), valores.get("bmi"), valores.get("waist_circumference"))
            + _de_presion(valores.get("ap_hi"), valores.get("ap_lo")))
