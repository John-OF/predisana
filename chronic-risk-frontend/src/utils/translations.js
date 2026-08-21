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
    high_cholesterol: "Colesterol Alto Diagnosticado",
    pregnancies: "Embarazos",
    skin_thickness: "Grosor Pliegue Cutáneo (mm)",
    diabetes_pedigree: "Función Pedigree Diabetes",

    // Antropométricos (hipertensión)
    weight: "Peso (kg)",
    waist_circumference: "Circunferencia de Cintura (cm)",

    // Cardiovascular
    ap_hi: "Presión Sistólica (mm Hg)",
    ap_lo: "Presión Diastólica (mm Hg)",
    cholesterol: "Colesterol (1=Normal a 3=Muy alto)",
    gluc: "Glucosa (1=Normal a 3=Muy alta)",
    smoke: "¿Fuma?",
    alco: "¿Consume alcohol?",
    active: "¿Actividad física habitual?",

    // Opciones
    Male: "Masculino",
    Female: "Femenino",
    current: "Fumador Actual",
    former: "Ex-Fumador",
    never: "Nunca ha fumado",
    ever: "Alguna vez",
    not_current: "No actual",
    
    // Enfermedades
    diabetes: "Diabetes Tipo 2",
    hipertension: "Hipertensión Arterial",
    cardiovascular: "Riesgo Cardiovascular"
};

export const getLabel = (key) => LABELS_ES[key] || key;