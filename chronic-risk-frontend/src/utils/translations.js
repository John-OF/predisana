export const LABELS_ES = {
    // Generales
    age: "Edad",
    gender: "Sexo",
    smoking_history: "Tabaquismo",
    weight: "Peso (kg)",
    height: "Talla (cm)",
    bmi: "Índice de Masa Corporal (IMC)",
    bmi_autodeclarado: "IMC autodeclarado",

    // Diagnósticos previos ("¿te lo ha dicho un médico?")
    heart_disease: "Enfermedad Cardiovascular Previa",
    hypertension: "Hipertensión Previa",
    high_cholesterol: "Colesterol Alto Diagnosticado",

    // Medidas y análisis (modo completo)
    waist_circumference: "Circunferencia de Cintura (cm)",
    ap_hi: "Presión Sistólica (mm Hg)",
    ap_lo: "Presión Diastólica (mm Hg)",
    total_cholesterol: "Colesterol Total (mg/dL)",
    hdl_cholesterol: "Colesterol HDL (mg/dL)",
    hba1c_level: "Hemoglobina Glicosilada (HbA1c, %)",
    blood_glucose_level: "Glucosa en Ayunas (mg/dL)",
    egfr: "Filtrado Glomerular (eGFR)",
    albumin_creatinine_ratio: "Albúmina/Creatinina en Orina (mg/g)",

    // Opciones
    Male: "Masculino",
    Female: "Femenino",
    current: "Fumador actual",
    former: "Exfumador",
    never: "Nunca ha fumado",

    // Modos
    simplificado: "Simplificado",
    completo: "Completo",

    // Enfermedades (`diabetes` es también una variable: el formulario la reetiqueta)
    diabetes: "Diabetes Tipo 2",
    hipertension: "Hipertensión Arterial",
    cardiovascular: "Enfermedad Cardiovascular"
};

export const getLabel = (key) => LABELS_ES[key] || key;

// Grupos que el formulario pide como una sola respuesta y el modelo recibe en columnas
// one-hot (gender_Male, smoking_history_former).
export const GRUPOS_ONE_HOT = ['gender', 'smoking_history'];

export const grupoOneHot = (feature) =>
    GRUPOS_ONE_HOT.find((g) => String(feature).startsWith(`${g}_`)) ?? null;

/** Etiqueta de una columna del modelo: las one-hot se leen como grupo y opción
 * ("Tabaquismo: Exfumador"); getLabel las dejaba con su clave cruda. */
export const getFeatureLabel = (feature) => {
    const grupo = grupoOneHot(feature);
    return grupo ? `${getLabel(grupo)}: ${getLabel(feature.slice(grupo.length + 1))}` : getLabel(feature);
};
