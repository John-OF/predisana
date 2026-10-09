// Ayuda de cada campo del simulador: el rango recomendado que se muestra bajo el input
// y su paso. El rango ACEPTADO (min/max) no vive aquí: lo sirve el backend en
// /config.ranges, que es el mismo que valida /predict.
//
// El paso tiene que admitir la precisión del dato real. «Caso virtual» carga una ficha
// del sintético, que lleva los decimales del real: la presión de NHANES es la media de
// tres lecturas (101,4), la talla viene de pulgadas (172,7) y la albúmina en orina
// lleva dos decimales. Con un paso de 1, el navegador marcaba el campo como inválido y
// bloqueaba el envío sin que el formulario dijera nada.
export const FIELD_HINTS = {
  age: { label: "18 - 100 años", step: 1 },
  weight: { label: "40 - 150 kg", step: 0.1 },
  height: { label: "145 - 200 cm", step: 0.1 },
  bmi: { label: "18.5 - 40", step: 0.1 },
  heart_disease: { label: "0 (No) - 1 (Sí)", step: 1 },
  hypertension: { label: "0 (No) - 1 (Sí)", step: 1 },
  diabetes: { label: "0 (No) - 1 (Sí)", step: 1 },
  high_cholesterol: { label: "0 (No) - 1 (Sí)", step: 1 },
  waist_circumference: { label: "60 - 120 cm", step: 0.1 },
  ap_hi: { label: "90 - 180 mmHg", step: 0.1 },
  ap_lo: { label: "60 - 120 mmHg", step: 0.1 },
  total_cholesterol: { label: "< 200 mg/dL deseable", step: 1 },
  hdl_cholesterol: { label: "> 40 (H) / > 50 (M) mg/dL", step: 1 },
  hba1c_level: { label: "4 - 9 %", step: 0.1 },
  blood_glucose_level: { label: "70 - 99 mg/dL en ayunas", step: 1 },
  egfr: { label: "≥ 60 normal", step: 0.1 },
  albumin_creatinine_ratio: { label: "< 30 mg/g normal", step: 0.01 },
  alt: { label: "< 35 U/L normal", step: 1 },
  default: { label: "Valor positivo", step: 1 }
};
