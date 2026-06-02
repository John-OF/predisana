# Diccionario de Datos: Hipertension

**Total de registros:** 353224

| Nombre del Campo | Tipo de Dato | Longitud | Descripción | Restricción | Ejemplo |
| --- | --- | --- | --- | --- | --- |
| age | DECIMAL | N/A | Edad del paciente en años. | Not Null, Rango: [0.1 - 93.0] | 50.0 |
| pregnancies | DECIMAL | N/A | Número de embarazos previos. | Not Null, Rango: [0.0 - 17.0] | 6.0 |
| glucose | DECIMAL | N/A | Nivel de glucosa en sangre (mg/dL). | Not Null, Rango: [10.4 - 2372.0] | 148.0 |
| blood_pressure | DECIMAL | N/A | Presión arterial (sistólica/diastólica unificada o predominante). | Not Null, Rango: [-150.0 - 16020.0] | 72.0 |
| skin_thickness | DECIMAL | N/A | Grosor del pliegue cutáneo del tríceps (mm). | Not Null, Rango: [0.0 - 99.0] | 35.0 |
| insulin | DECIMAL | N/A | Nivel de insulina sérica (mu U/ml). | Not Null, Rango: [0.0 - 846.0] | 0.0 |
| bmi | DECIMAL | N/A | Índice de Masa Corporal (kg/m²). | Not Null, Rango: [1.0 - 298.7] | 33.6 |
| diabetes_pedigree | DECIMAL | N/A | Función de pedigrí de diabetes (historial genético). | Not Null, Rango: [0.0 - 2.4] | 0.627 |
| hypertension | DECIMAL | N/A | Historial de hipertensión (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0.0 |
| heart_disease | INT | N/A | Historial de enfermedad cardíaca (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| hba1c_level | DECIMAL | N/A | Nivel de Hemoglobina Glicosilada (%). | Not Null, Rango: [0.9 - 17.2] | 5.8 |
| blood_glucose_level | DECIMAL | N/A | Nivel de glucosa en sangre (copia para compatibilidad). | Not Null, Rango: [10.4 - 2372.0] | 148.0 |
| gender_Female | INT | N/A | Variable dummy: Género Femenino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| gender_Male | INT | N/A | Variable dummy: Género Masculino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| smoking_history_current | INT | N/A | Fumador actual. | Not Null, Binario (0, 1) | 0 |
| smoking_history_former | INT | N/A | Ex-fumador. | Not Null, Binario (0, 1) | 0 |
| smoking_history_never | INT | N/A | Nunca ha fumado. | Not Null, Binario (0, 1) | 0 |
| smoking_history_ever | INT | N/A | Alguna vez ha fumado. | Not Null, Binario (0, 1) | 0 |
| smoking_history_not current | INT | N/A | No fuma actualmente. | Not Null, Binario (0, 1) | 0 |
| target | INT | N/A | Variable objetivo (Clase a predecir: 0=Negativo, 1=Positivo). | Not Null, Binario (0, 1) | 0 |
