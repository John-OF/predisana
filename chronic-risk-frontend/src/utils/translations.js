export const LABELS_ES = {
    // Generales
    age: "Edad",
    gender: "Género",
    bmi: "Índice de Masa Corporal (BMI)",
    smoking_history: "Historial de Tabaquismo",
    
    // Clínicos
    glucose: "Nivel de Glucosa (mg/dL)",
    blood_glucose_level: "Glucosa en Sangre",
    blood_pressure: "Presión Arterial (mm Hg)",
    insulin: "Insulina (mu U/ml)",
    hba1c_level: "Hemoglobina Glicosilada (HbA1c)",
    heart_disease: "Enfermedad Cardíaca Previa",
    hypertension: "Hipertensión Previa",
    pregnancies: "Embarazos",
    skin_thickness: "Grosor Pliegue Cutáneo (mm)",
    diabetes_pedigree: "Función Pedigree Diabetes",
    
    // Opciones
    Male: "Masculino",
    Female: "Femenino",
    current: "Fumador Actual",
    former: "Ex-Fumador",
    never: "Nunca ha fumado",
    ever: "Alguna vez",
    "not current": "No actual",
    
    // Enfermedades
    diabetes: "Diabetes Tipo 2",
    hipertension: "Hipertensión Arterial",
    obesidad: "Obesidad",
    cardiovascular: "Riesgo Cardiovascular"
};

export const getLabel = (key) => LABELS_ES[key] || key;