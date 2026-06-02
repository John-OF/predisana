import { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, ProgressBar, Nav, Spinner, Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { getConfig, predictRisk, getSyntheticCase } from '../services/api';
import { saveHistoryEntry } from '../services/historyService';
import { getLabel } from '../utils/translations';
import Swal from 'sweetalert2';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

const DISEASES = ['diabetes', 'hipertension', 'obesidad', 'cardiovascular'];

// --- CONFIGURACIÓN DE LÍMITES CLÍNICOS REALISTAS ---
const CLINICAL_LIMITS = {
    age: { min: 1, max: 120, label: "1 - 100 años", step: 1 },
    pregnancies: { min: 0, max: 20, label: "0 - 10", step: 1 },
    glucose: { min: 40, max: 500, label: "70 - 200 mg/dL", step: 1 },
    blood_pressure: { min: 50, max: 300, label: "90 - 180 mmHg", step: 1 },
    skin_thickness: { min: 0, max: 99, label: "10 - 50 mm", step: 1 },
    insulin: { min: 0, max: 900, label: "15 - 276 mu U/ml", step: 1 },
    bmi: { min: 10, max: 90, label: "18.5 - 40", step: 0.1 },
    diabetes_pedigree: { min: 0, max: 3, label: "0.08 - 2.42", step: 0.01 },
    hba1c_level: { min: 3, max: 15, label: "4 - 9 %", step: 0.1 },
    heart_disease: { min: 0, max: 1, label: "0 (No) - 1 (Sí)", step: 1 }, 
    hypertension: { min: 0, max: 1, label: "0 (No) - 1 (Sí)", step: 1 },
    default: { min: 0, max: 1000, label: "Valor positivo", step: 1 }
};

// --- DICCIONARIO EDUCATIVO MEJORADO (Más intuitivo) ---
const VARIABLE_DESCRIPTIONS = {
    age: "La edad influye directamente en el riesgo acumulado. A mayor edad, mayor vigilancia requerida.",
    gender: "Factor biológico que puede predisponer a ciertas condiciones hormonales o cardiovasculares.",
    bmi: "Índice de Masa Corporal. Un valor superior a 25 indica sobrepeso; superior a 30, obesidad.",
    glucose: "Nivel de azúcar en sangre. Lo ideal en ayunas es entre 70 y 100 mg/dL.",
    blood_glucose_level: "Nivel actual de azúcar en sangre.",
    blood_pressure: "Presión arterial. Valores constantes sobre 130/80 mmHg aumentan el riesgo cardíaco.",
    insulin: "Hormona que regula el azúcar. Niveles muy altos sugieren que tu cuerpo resiste su efecto.",
    hba1c_level: "Hemoglobina Glicosilada. Es tu 'promedio' de azúcar de los últimos 3 meses. Ideal: menos de 5.7%.",
    pregnancies: "Número de embarazos. Importante para evaluar antecedentes de diabetes gestacional.",
    skin_thickness: "Medida del pliegue del brazo. Ayuda a estimar la grasa corporal real.",
    diabetes_pedigree: "Puntaje basado en tus antecedentes familiares. Indica predisposición genética.",
    smoking_history: "Fumar daña las arterias y el corazón. Es el factor de riesgo modificable más crítico.",
    heart_disease: "Indica si ya has tenido diagnósticos cardíacos previos.",
    hypertension: "Indica si ya has sido diagnosticado previamente con presión alta.",
    default: "Variable clínica utilizada por la Inteligencia Artificial."
};

// --- COMPONENTE AUXILIAR PARA EL TOOLTIP ---
const InfoIcon = ({ variableKey }) => {
    const text = VARIABLE_DESCRIPTIONS[variableKey] || VARIABLE_DESCRIPTIONS.default;
    
    return (
        <OverlayTrigger
            placement="auto"
            overlay={<Tooltip id={`tooltip-${variableKey}`}>{text}</Tooltip>}
        >
            <span 
                className="ms-2 badge rounded-pill bg-info text-dark border border-info"
                style={{ cursor: 'help', fontSize: '0.65rem', verticalAlign: 'text-top' }}
            >
                ?
            </span>
        </OverlayTrigger>
    );
};

const SHAP_LABELS_ES = {
  blood_pressure: "Presión Arterial (mm Hg)",
  glucose: "Nivel de Glucosa (mg/dL)",
  blood_glucose_level: "Glucosa en Sangre (mg/dL)",
  hba1c_level: "Hemoglobina Glicosilada (HbA1c)",
  bmi: "Índice de Masa Corporal (IMC)",
  age: "Edad",
  insulin: "Insulina (µU/mL)",
  skin_thickness: "Grosor Pliegue Cutáneo (mm)",
  diabetes_pedigree: "Función Pedigree Diabetes",
  pregnancies: "Embarazos",
  heart_disease: "Enfermedad Cardíaca Previa",
  hypertension: "Hipertensión Previa",

  smoking_history_current: "Tabaquismo: Actual",
  smoking_history_not_current: "Tabaquismo: No actual",
  smoking_history_former: "Tabaquismo: Exfumador",
  smoking_history_never: "Tabaquismo: Nunca",
  smoking_history_ever: "Tabaquismo: Alguna vez",

  gender_Male: "Género: Masculino",
  gender_Female: "Género: Femenino",
};

const labelES = (feat) => {
  const key = String(feat || "").trim().replace(/\s+/g, "_"); // espacios -> _
  return SHAP_LABELS_ES[key] || SHAP_LABELS_ES[String(feat || "").trim()] || getLabel(feat) || feat;
};


const Simulacion = () => {
    const [selectedDisease, setSelectedDisease] = useState('diabetes');
    const [config, setConfig] = useState(null);
    const [formData, setFormData] = useState({});
    const [loading, setLoading] = useState(false);
    
    // Estado para resultados
    const [currentResult, setCurrentResult] = useState(null);
    const [baseResult, setBaseResult] = useState(null);
    const [error, setError] = useState(null);
    const [baseCase, setBaseCase] = useState(null); 
    // baseCase = snapshot completo (inputs + result + timestamp) del escenario fijado


    // 1. Cargar configuración
    useEffect(() => {
        const fetchConfig = async () => {
            setLoading(true);
            setError(null);
            setCurrentResult(null);
            setBaseResult(null); 
            setFormData({}); 
            setBaseCase(null);
            try {
                const { data } = await getConfig(selectedDisease);
                setConfig(data);
                
                // Inicializar valores por defecto de categoricas para evitar selects vacíos
                const initialData = {};
                if (data.categoricals) {
                    Object.keys(data.categoricals).forEach(cat => {
                        initialData[cat] = data.categoricals[cat][0];
                    });
                }
                setFormData(initialData);
                
            } catch (err) {
                console.error(err);
                setError("Error cargando la configuración del modelo.");
            } finally {
                setLoading(false);
            }
        };
        fetchConfig();
    }, [selectedDisease]);

    // 2. Generar Caso Sintético (CORREGIDO Y MEJORADO)
    const handleGenerateSynthetic = async () => {
        setLoading(true);
        try {
            const { data } = await getSyntheticCase(selectedDisease);
            
            // Creamos un objeto limpio
            const cleanData = {};

            // LISTAS DE VARIABLES SEGÚN TIPO PARA LIMPIEZA
            const integers = ['age', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'hypertension', 'heart_disease'];
            const floats_1 = ['bmi', 'hba1c_level']; // 1 decimal
            const floats_2 = ['diabetes_pedigree']; // 2 decimales

            // Procesar valores numéricos
            Object.keys(data).forEach(key => {
                let val = data[key];
                
                if (integers.includes(key)) {
                    cleanData[key] = Math.round(val);
                } else if (floats_1.includes(key)) {
                    cleanData[key] = parseFloat(Number(val).toFixed(1));
                } else if (floats_2.includes(key)) {
                    cleanData[key] = parseFloat(Number(val).toFixed(2));
                } else {
                    // Si no está en las listas, lo pasamos tal cual (ej. blood_glucose_level)
                     // Si parece entero pero viene como float, redondeamos por seguridad si no es key especial
                    if (typeof val === 'number' && !key.includes('pedigree')) {
                        cleanData[key] = Math.round(val);
                    } else {
                        cleanData[key] = val;
                    }
                }
            });
            
            // Procesar categóricas (mapeo inverso de One-Hot a Label)
            if (config.categoricals) {
                Object.keys(config.categoricals).forEach(cat => {
                    const options = config.categoricals[cat];
                    // Buscamos cuál opción tiene el valor 1 en el one-hot encoding
                    // Ej: busca si data['gender_Male'] == 1
                    const activeOption = options.find(opt => data[`${cat}_${opt}`] === 1);
                    
                    if (activeOption) {
                        cleanData[cat] = activeOption;
                    } else {
                        // Fallback: si no encuentra (ej. caso base), poner el primero
                        cleanData[cat] = options[0];
                    }
                });
            }
            
            setFormData(prev => ({...prev, ...cleanData}));
            setCurrentResult(null); 
            
            const Toast = Swal.mixin({ toast: true, position: 'top-end', showConfirmButton: false, timer: 3000 });
            Toast.fire({ icon: 'success', title: 'Caso virtual generado correctamente' });

        } catch (err) {
            console.error(err);
            setError("No se pudo generar el caso sintético.");
        } finally {
            setLoading(false);
        }
    };

    // 3. Manejar cambios en inputs
    const handleChange = (e) => {
        const { name, value, type } = e.target;
        
        if (currentResult && !baseResult) {
            setCurrentResult(null); 
        }

        if (type === 'number') {
            const numValue = parseFloat(value);
            // Permitir borrar el campo (string vacío) o valores válidos
            if (value === '') {
                 setFormData(prev => ({ ...prev, [name]: '' }));
                 return;
            }
            setFormData(prev => ({ ...prev, [name]: numValue }));
        } else {
            setFormData(prev => ({ ...prev, [name]: value }));
        }
    };

    const preparePayload = () => {
        const payload = { ...formData };

        // Forzar consistencia entre glucosas (compatibilidad)
        if (payload.glucose !== undefined) {
        payload.blood_glucose_level = payload.glucose;
        }

        if (config.categoricals) {
            Object.keys(config.categoricals).forEach(catName => {
                const selectedValue = payload[catName];
                delete payload[catName]; // Borramos la variable categórica "humana"
                
                // Generamos las variables dummy que necesita el modelo (ej. gender_Male)
                config.features.forEach(feat => {
                    if (feat.startsWith(`${catName}_`)) {
                        const option = feat.replace(`${catName}_`, '');
                        payload[feat] = (option === selectedValue) ? 1 : 0;
                    }
                });
            });
        }
        return payload;
    };

    const makeId = () => {
        if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
        return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
    };

    const buildRunSnapshot = (inputs, apiData) => ({
        id: makeId(),
        timestamp: Date.now(),
        inputs: { ...inputs }, // aquí guardamos tu formData completo (incluye sintético)
        result: {
            probability: apiData?.probability,
            prediction: apiData?.prediction,
        },
        top_features: apiData?.top_features ?? null,
    });

    const saveSinglePredictionToHistory = (apiData) => {
    const entry = {
        type: "single",
        id: makeId(),
        timestamp: Date.now(),
        disease: selectedDisease,
        inputs: { ...formData },
        result: {
        probability: apiData?.probability,
        prediction: apiData?.prediction,
        },
        top_features: apiData?.top_features ?? null,
        notes: "",
    };
    saveHistoryEntry(entry);
    };

    const saveComparisonToHistory = (newApiData) => {
        if (!baseCase) return;

        const newSnap = buildRunSnapshot(formData, newApiData);

        const basePct = (baseCase.result.probability ?? 0) * 100;
        const newPct = (newApiData.probability ?? 0) * 100;
        const diff = newPct - basePct;

        const entry = {
            type: "comparison",
            id: makeId(),
            timestamp: Date.now(),
            disease: selectedDisease,
            base: {
            timestamp: baseCase.timestamp,
            inputs: { ...baseCase.inputs },
            result: { ...baseCase.result },
            },
            scenarios: [
            {
                timestamp: newSnap.timestamp,
                inputs: { ...newSnap.inputs },
                result: { ...newSnap.result },
            },
            ],
            delta: {
            probability_points: Number(diff.toFixed(1)), // diferencia en puntos porcentuales
            direction: diff < 0 ? "decrease" : diff > 0 ? "increase" : "same",
            },
            top_features_new: newApiData?.top_features ?? null,
            notes: "",
        };

        saveHistoryEntry(entry);
    };


    // 4. Enviar predicción
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const payload = preparePayload();
            const { data } = await predictRisk(selectedDisease, payload);

            setCurrentResult(data);

            // ✅ Guardar historial
            if (baseResult) {
            // estamos en modo comparación: guardar base + nuevo + delta
            saveComparisonToHistory(data);
            } else {
            // predicción normal
            saveSinglePredictionToHistory(data);
            }

            if (!baseResult && data.prediction === 1) {
            Swal.fire({
                icon: 'warning',
                title: 'Atención',
                text: `El modelo ha detectado un riesgo elevado (${(data.probability * 100).toFixed(1)}%)`,
                confirmButtonColor: '#d33'
            });
            }
        } catch (err) {
            console.error(err);
            setError("Error al procesar la predicción. Revisa que todos los campos numéricos tengan valores.");
        } finally {
            setLoading(false);
        }
    };


    // 5. Comparación
    const handleSetBaseCase = () => {
        // Guardamos el escenario base completo (inputs + resultado)
        setBaseCase({
            timestamp: Date.now(),
            inputs: { ...formData },
            result: {
                probability: currentResult?.probability,
                prediction: currentResult?.prediction,
            },
        });

        setBaseResult(currentResult);
        setCurrentResult(null);

        Swal.fire({
            icon: 'success',
            title: 'Escenario Base Fijado',
            text: 'Ahora modifica las variables (ej. baja el peso) y vuelve a calcular.'
        });
    };

    

    const handleResetComparison = () => {
        setBaseCase(null);
        setBaseResult(null);
        setCurrentResult(null);
    };


    // Renderizado de campos numéricos
    const renderNumberInput = (feat) => {

        if (feat === "blood_glucose_level") return null;

        const isCategoricalPart = config.categoricals && Object.keys(config.categoricals).some(cat => feat.startsWith(cat + "_"));
        if (isCategoricalPart) return null;

        // Ocultar campo pregnancies si el género seleccionado es Male
        if (feat === 'pregnancies' && formData['gender'] === 'Male') return null;

        const limits = CLINICAL_LIMITS[feat] || CLINICAL_LIMITS.default;

        return (
            <Form.Group className="mb-3" key={feat}>
                <Form.Label className="d-flex align-items-center justify-content-between">
                    <span>{getLabel(feat)}</span>
                    <InfoIcon variableKey={feat} />
                </Form.Label>
                <Form.Control 
                    type="number" 
                    name={feat}
                    value={formData[feat] !== undefined ? formData[feat] : ''}
                    onChange={handleChange}
                    min={limits.min}
                    max={limits.max}
                    step={limits.step || "any"}
                    placeholder={`Rango: ${limits.min} - ${limits.max}`}
                    required
                />
                <Form.Text className="text-muted d-block text-end small">
                   Recomendado: {limits.label}
                </Form.Text>
            </Form.Group>
        );
    };

    // Gráfico Comparativo
    const renderComparisonChart = () => {
        if (!baseResult || !currentResult) return null;

        const data = [
            {
                name: 'Probabilidad de Riesgo',
                Base: (baseResult.probability * 100).toFixed(1),
                Nuevo: (currentResult.probability * 100).toFixed(1),
            }
        ];

        const diff = (currentResult.probability * 100) - (baseResult.probability * 100);
        const isImprovement = diff < 0;

        return (
            <div className="mt-4 animate__animated animate__fadeIn">
                <hr />
                <h5 className="mb-3">⚖️ Comparativa de Escenarios</h5>
                
                <Alert variant={isImprovement ? 'success' : 'warning'}>
                    <strong>Conclusión: </strong> 
                    {isImprovement 
                        ? `¡Excelente! Con estos cambios, tu riesgo estimado se reduciría un ${Math.abs(diff).toFixed(1)}%.`
                        : `Cuidado. Estos cambios aumentarían tu riesgo estimado en un ${Math.abs(diff).toFixed(1)}%.`
                    }
                </Alert>

                <div style={{ width: '100%', height: 250 }}>
                    <ResponsiveContainer>
                        <BarChart data={data} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" domain={[0, 100]} unit="%" />
                            <YAxis type="category" dataKey="name" hide />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="Base" fill="#6c757d" name="Escenario Inicial" barSize={30} />
                            <Bar dataKey="Nuevo" fill={isImprovement ? "#198754" : "#dc3545"} name="Escenario Simulado" barSize={30} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    };

    // Gráfico SHAP (Top 5)
    const renderShapChart = () => {
        // 1) Filtrar para no mostrar la glucosa duplicada
        const feats = (currentResult?.top_features || []).filter(f => f.feature !== "blood_glucose_level");
        if (feats.length === 0) return null;

        // 2) Escala para barra azul (importancia = |SHAP|)
        const maxAbs = Math.max(...feats.map(f => Math.abs(Number(f.shap))), 1e-6);

        return (
            <div className="mt-4 animate__animated animate__fadeIn">
            <hr />
            <h6 className="text-muted mb-2">🔍 Variables más influyentes (IA)</h6>

            <p className="small text-muted mb-3">
                Las barras azules indican la importancia (SHAP). Flecha indica si la variable <b>aumenta o reduce</b> la salida del{" "}
                <b>modelo IA</b>. La probabilidad final puede cambiar por reglas clínicas.
            </p>

            <div className="d-flex flex-column gap-3">
                {feats.map((f, idx) => {
                const shapVal = Number(f.shap);
                const absVal = Math.abs(shapVal);
                const percent = Math.min((absVal / maxAbs) * 100, 100);

                const directionText = shapVal >= 0 ? "↑ Aumenta (modelo)" : "↓ Reduce (modelo)";
                const directionClass = shapVal >= 0 ? "text-danger" : "text-success";

                return (
                    <div key={idx}>
                    <div className="d-flex justify-content-between align-items-center mb-1">
                        <small className="fw-semibold">{labelES(f.feature)}</small>
                        <small className={directionClass} style={{ fontWeight: 600 }}>
                        {directionText}
                        </small>
                    </div>

                    <ProgressBar
                        now={percent}
                        striped
                        animated
                        variant="primary"
                        style={{ height: "12px", borderRadius: "10px", overflow: "hidden" }}
                    />
                    </div>
                );
                })}
            </div>
            </div>
        );
    };

    return (
        <Container className="py-4">
            <h2 className="mb-4 text-primary fw-bold">Simulador de Riesgo Clínico</h2>
            
            {/* TABS DE SELECCIÓN DE ENFERMEDAD */}
            <Card className="mb-4 shadow-sm border-0">
                <Card.Header className="bg-white">
                    <Nav variant="tabs" defaultActiveKey="diabetes" className="border-bottom-0">
                        {DISEASES.map(d => (
                            <Nav.Item key={d}>
                                <Nav.Link 
                                    active={selectedDisease === d}
                                    onClick={() => setSelectedDisease(d)}
                                    className={`text-capitalize px-4 ${selectedDisease === d ? 'fw-bold' : 'text-muted'}`}
                                >
                                    {getLabel(d)}
                                </Nav.Link>
                            </Nav.Item>
                        ))}
                    </Nav>
                </Card.Header>
            </Card>

            {error && <Alert variant="danger">{error}</Alert>}

            {loading && !config ? (
                <div className="text-center py-5">
                    <Spinner animation="border" variant="primary" />
                    <p className="mt-2 text-muted">Cargando modelos de IA...</p>
                </div>
            ) : (
                config && (
                    <Row>
                        {/* COLUMNA IZQUIERDA: FORMULARIO */}
                        <Col md={7}>
                            <Card className="shadow-sm border-0 h-100">
                                <Card.Body className="p-4">
                                    <div className="d-flex justify-content-between align-items-center mb-4">
                                        <h5 className="mb-0 text-dark">📝 Ingreso de Datos</h5>
                                        <div className="d-flex gap-2">
                                            {baseResult && (
                                                <Button variant="outline-secondary" size="sm" onClick={handleResetComparison}>
                                                    🔄 Reiniciar
                                                </Button>
                                            )}
                                            <OverlayTrigger placement="top" overlay={<Tooltip>Genera un paciente ficticio con datos realistas</Tooltip>}>
                                                <Button 
                                                    variant="outline-primary" 
                                                    size="sm" 
                                                    onClick={handleGenerateSynthetic}
                                                    disabled={loading || baseResult}
                                                >
                                                    🎲 Caso Virtual
                                                </Button>
                                            </OverlayTrigger>
                                        </div>
                                    </div>

                                    {baseResult && (
                                        <Alert variant="info" className="py-2 mb-4">
                                            <small><strong>Modo Comparación:</strong> Ahora modifica los valores del formulario (ej. reduce el peso) y recalcula para ver el impacto.</small>
                                        </Alert>
                                    )}

                                    <Form onSubmit={handleSubmit}>
                                        {/* 1. CAMPOS CATEGÓRICOS (Dropdowns) */}
                                        {config.categoricals && Object.keys(config.categoricals).map(cat => (
                                            <Form.Group className="mb-3" key={cat}>
                                                <Form.Label className="d-flex align-items-center justify-content-between">
                                                    <span>{getLabel(cat)}</span>
                                                    <InfoIcon variableKey={cat} />
                                                </Form.Label>
                                                <Form.Select 
                                                    name={cat} 
                                                    value={formData[cat] || ''} 
                                                    onChange={handleChange}
                                                    className="bg-light"
                                                >
                                                    {config.categoricals[cat].map(opt => (
                                                        <option key={opt} value={opt}>{getLabel(opt)}</option>
                                                    ))}
                                                </Form.Select>
                                            </Form.Group>
                                        ))}

                                        <hr className="my-4 text-muted" />

                                        {/* 2. CAMPOS NUMÉRICOS */}
                                        <Row>
                                            {config.features.map(feat => {
                                                const input = renderNumberInput(feat);
                                                return input ? <Col md={6} key={feat}>{input}</Col> : null;
                                            })}
                                        </Row>

                                        <div className="d-grid gap-2 mt-4">
                                            <Button 
                                                variant={baseResult ? "success" : "primary"} 
                                                size="lg" 
                                                type="submit" 
                                                disabled={loading}
                                                className="fw-bold"
                                            >
                                                {loading ? <Spinner as="span" animation="border" size="sm" /> : (baseResult ? '🔍 Comparar Nuevo Escenario' : '🚀 Calcular Riesgo')}
                                            </Button>
                                        </div>
                                    </Form>
                                </Card.Body>
                            </Card>
                        </Col>

                        {/* COLUMNA DERECHA: RESULTADOS */}
                        <Col md={5}>
                            <Card className={`text-center shadow border-0 h-100 ${currentResult ? 'bg-white' : 'bg-light'}`}>
                                <Card.Body className="d-flex flex-column justify-content-center p-4">
                                    {!currentResult && !baseResult && (
                                        <div className="text-muted opacity-50">
                                            <div style={{ fontSize: '5rem' }}>🩺</div>
                                            <h5 className="mt-3">Esperando datos...</h5>
                                            <p className="small">Completa el formulario o usa el botón "Caso Virtual" para comenzar.</p>
                                        </div>
                                    )}

                                    {currentResult && (
                                        <div className="w-100 animate__animated animate__fadeIn">
                                            <h5 className="text-muted mb-4">
                                                {baseResult ? "Nuevo Escenario Simulado" : "Resultado del Análisis"}
                                            </h5>
                                            
                                            <div className="mb-4 position-relative">
                                                <h1 className={`display-3 fw-bold ${currentResult.prediction === 1 ? 'text-danger' : 'text-success'}`}>
                                                    {(currentResult.probability * 100).toFixed(1)}%
                                                </h1>
                                                <p className="text-muted small">Probabilidad de riesgo estimada</p>
                                                
                                                <ProgressBar
                                                    now={currentResult.probability * 100}
                                                    variant={currentResult.prediction === 1 ? 'danger' : 'success'}
                                                    striped
                                                    animated
                                                    style={{ height: '12px', borderRadius: '10px', overflow: 'hidden' }}
                                                    className="mt-2 risk-progress"
                                                />

                                            </div>

                                            <Card bg={currentResult.prediction === 1 ? 'danger' : 'success'} text="white" className="mb-4 shadow-sm">
                                                <Card.Body className="py-2">
                                                    <Card.Title className="mb-0 fw-bold">
                                                        {currentResult.prediction === 1 ? 'RIESGO ALTO' : 'RIESGO BAJO'}
                                                    </Card.Title>
                                                </Card.Body>
                                            </Card>

                                            {currentResult && renderShapChart()}

                                            {!baseResult && (
                                                <div className="d-grid">
                                                     <Button 
                                                        variant="outline-dark" 
                                                        className="mt-2"
                                                        onClick={handleSetBaseCase}
                                                    >
                                                        ⚖️ Comparar con otro escenario
                                                    </Button>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {renderComparisonChart()}

                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                )
            )}
        </Container>
    );
};

export default Simulacion;