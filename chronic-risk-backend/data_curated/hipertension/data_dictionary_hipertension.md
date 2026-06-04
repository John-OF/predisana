# Diccionario de Datos: Hipertension

**Total de registros:** 4363

| Nombre del Campo | Tipo de Dato | Longitud | Descripción | Restricción | Ejemplo |
| --- | --- | --- | --- | --- | --- |
| age | INT | N/A | Edad del paciente en años. | Not Null, Rango: [4.0 - 93.0] | 41 |
| bmi | DECIMAL | N/A | Índice de Masa Corporal (kg/m²). | Not Null, Rango: [12.6 - 60.5] | 32.889389 |
| weight | DECIMAL | N/A | Peso del paciente. | Not Null, Rango: [26.8 - 168.8] | 74.55 |
| waist_circumference | DECIMAL | N/A | Circunferencia de cintura (cm). | Not Null, Rango: [58.8 - 189.3] | 97.5 |
| blood_pressure | DECIMAL | N/A | Presión arterial (sistólica/diastólica unificada o predominante). | Not Null, Rango: [80.0 - 200.0] | 107.0 |
| glucose | DECIMAL | N/A | Nivel de glucosa en sangre (mg/dL). | Not Null, Rango: [41.0 - 480.0] | 92.0 |
| hba1c_level | DECIMAL | N/A | Nivel de Hemoglobina Glicosilada (%). | Not Null, Rango: [3.9 - 17.2] | 5.2 |
| cholesterol_total | INT | N/A | Colesterol total en sangre (mg/dL). | Not Null, Rango: [40.0 - 681.0] | 139 |
| hdl | INT | N/A | Colesterol HDL ('bueno') (mg/dL). | Not Null, Rango: [9.0 - 279.0] | 34 |
| ldl | DECIMAL | N/A | Colesterol LDL ('malo') (mg/dL). | Not Null, Rango: [11.1 - 303.0] | 86.0 |
| triglycerides | INT | N/A | Triglicéridos en sangre (mg/dL). | Not Null, Rango: [23.0 - 1320.0] | 123 |
| insulin | DECIMAL | N/A | Nivel de insulina sérica (mu U/ml). | Not Null, Rango: [0.6 - 264.1] | 4.0 |
| gender_Male | INT | N/A | Variable dummy: Género Masculino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 0 |
| gender_Female | INT | N/A | Variable dummy: Género Femenino (0=No, 1=Sí). | Not Null, Binario (0, 1) | 1 |
| target | INT | N/A | Variable objetivo (Clase a predecir: 0=Negativo, 1=Positivo). | Not Null, Binario (0, 1) | 1 |
