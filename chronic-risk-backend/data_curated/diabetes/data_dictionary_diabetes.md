# Diccionario de Datos: Diabetes

**Total de registros:** 100000

| Nombre del Campo | Tipo de Dato | Longitud | Descripción | Restricción | Ejemplo |
| --- | --- | --- | --- | --- | --- |
| age | DECIMAL | N/A | Edad del paciente en años. | Not Null, Rango: [0.1 - 80.0] | 80.0 |
| bmi | DECIMAL | N/A | Índice de Masa Corporal (kg/m²). | Not Null, Rango: [10.0 - 69.7] | 25.19 |
| hba1c_level | DECIMAL | N/A | Nivel de Hemoglobina Glicosilada (%). | Not Null, Rango: [3.5 - 9.0] | 6.6 |
| blood_glucose_level | INT | N/A | Nivel de glucosa en sangre (copia para compatibilidad). | Not Null, Rango: [80.0 - 300.0] | 140 |
| hypertension | INT | N/A | Historial de hipertensión (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| heart_disease | INT | N/A | Historial de enfermedad cardíaca (0=No, 1=Sí). | Not Null, Binario (0, 1) | 1 |
| gender_Female | INT | N/A | Variable dummy: Género Femenino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 1 |
| gender_Male | INT | N/A | Variable dummy: Género Masculino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| smoking_history_never | INT | N/A | Nunca ha fumado. | Not Null, Binario (0, 1) | 1 |
| smoking_history_current | INT | N/A | Fumador actual. | Not Null, Binario (0, 1) | 0 |
| smoking_history_former | INT | N/A | Ex-fumador. | Not Null, Binario (0, 1) | 0 |
| smoking_history_ever | INT | N/A | Alguna vez ha fumado. | Not Null, Binario (0, 1) | 0 |
| smoking_history_not_current | INT | N/A | No fuma actualmente. | Not Null, Binario (0, 1) | 0 |
| target | INT | N/A | Variable objetivo (Clase a predecir: 0=Negativo, 1=Positivo). | Not Null, Binario (0, 1) | 0 |
