import { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Alert, Nav, Spinner, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { getConfig, predictRisk, getSyntheticCase } from '../services/api';
import { saveHistoryEntry } from '../services/historyService';
import { getLabel } from '../utils/translations';
import Swal from 'sweetalert2';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  ArrowUpShort, ArrowDownShort, PencilSquare, ArrowCounterclockwise, Dice5Fill,
  RocketTakeoffFill, HeartPulse, ArrowLeftRight, Search, BarChartSteps, ClipboardData,
  ArrowRepeat, CpuFill,
} from 'react-bootstrap-icons';

const DISEASES = ['diabetes', 'hipertension', 'cardiovascular'];

const MODEL_LABELS = {
  logistic_regression: 'Regresión Logística',
  logreg: 'Regresión Logística',
  random_forest: 'Random Forest',
  lightgbm: 'LightGBM',
  xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');

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
  blood_glucose_level: { min: 40, max: 500, label: "70 - 200 mg/dL", step: 1 },
  weight: { min: 30, max: 250, label: "40 - 150 kg", step: 0.1 },
  waist_circumference: { min: 40, max: 200, label: "60 - 120 cm", step: 0.1 },
  ap_hi: { min: 70, max: 250, label: "90 - 180 mmHg", step: 1 },
  ap_lo: { min: 40, max: 150, label: "60 - 120 mmHg", step: 1 },
  cholesterol: { min: 1, max: 3, label: "1 (Normal) - 3 (Muy alto)", step: 1 },
  gluc: { min: 1, max: 3, label: "1 (Normal) - 3 (Muy alta)", step: 1 },
  smoke: { min: 0, max: 1, label: "0 (No) - 1 (Sí)", step: 1 },
  alco: { min: 0, max: 1, label: "0 (No) - 1 (Sí)", step: 1 },
  active: { min: 0, max: 1, label: "0 (No) - 1 (Sí)", step: 1 },
  default: { min: 0, max: 1000, label: "Valor positivo", step: 1 }
};

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
  weight: "Tu peso corporal en kilogramos.",
  waist_circumference: "Contorno de cintura en cm. Refleja la grasa abdominal, clave en el riesgo metabólico.",
  ap_hi: "Presión sistólica (la 'alta'): el primer número al medir la presión.",
  ap_lo: "Presión diastólica (la 'baja'): el segundo número al medir la presión.",
  cholesterol: "Nivel de colesterol según tu médico: 1=Normal, 2=Elevado, 3=Muy alto.",
  gluc: "Nivel de glucosa según tu médico: 1=Normal, 2=Elevada, 3=Muy alta.",
  smoke: "Indica si fumas actualmente (0=No, 1=Sí).",
  alco: "Indica si consumes alcohol habitualmente (0=No, 1=Sí).",
  active: "Indica si realizas actividad física habitual (0=No, 1=Sí).",
  default: "Variable clínica utilizada por la Inteligencia Artificial."
};

const InfoIcon = ({ variableKey }) => {
  const text = VARIABLE_DESCRIPTIONS[variableKey] || VARIABLE_DESCRIPTIONS.default;
  return (
    <OverlayTrigger placement="auto" overlay={<Tooltip id={`tooltip-${variableKey}`}>{text}</Tooltip>}>
      <span
        className="ms-2 badge rounded-pill"
        style={{ cursor: 'help', fontSize: '0.65rem', verticalAlign: 'text-top', background: 'var(--halo)', color: 'var(--accent)' }}
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
  weight: "Peso (kg)",
  waist_circumference: "Cintura (cm)",
  ap_hi: "Presión Sistólica",
  ap_lo: "Presión Diastólica",
  cholesterol: "Colesterol (1-3)",
  gluc: "Glucosa (1-3)",
  smoke: "Tabaquismo",
  alco: "Alcohol",
  active: "Actividad física",
  smoking_history_current: "Tabaquismo: Actual",
  smoking_history_not_current: "Tabaquismo: No actual",
  smoking_history_former: "Tabaquismo: Exfumador",
  smoking_history_never: "Tabaquismo: Nunca",
  smoking_history_ever: "Tabaquismo: Alguna vez",
  gender_Male: "Género: Masculino",
  gender_Female: "Género: Femenino",
};

const labelES = (feat) => {
  const key = String(feat || "").trim().replace(/\s+/g, "_");
  return SHAP_LABELS_ES[key] || SHAP_LABELS_ES[String(feat || "").trim()] || getLabel(feat) || feat;
};

// Banda de riesgo serena (no alarmista): bajo / moderado / alto
const riskBand = (pct) => {
  if (pct < 33) return { key: 'low', label: 'Riesgo bajo', color: 'var(--risk-low)', bg: 'var(--risk-low-bg)' };
  if (pct < 66) return { key: 'mid', label: 'Riesgo moderado', color: 'var(--risk-mid)', bg: 'var(--risk-mid-bg)' };
  return { key: 'high', label: 'Riesgo alto', color: 'var(--risk-high)', bg: 'var(--risk-high-bg)' };
};

// Medidor circular
const Gauge = ({ pct }) => {
  const R = 84;
  const C = 2 * Math.PI * R; // ≈ 528
  const offset = C - (C * pct) / 100;
  const band = riskBand(pct);
  return (
    <div className="ps-gauge-wrap">
      <div className="ps-gauge">
        <svg width="200" height="200">
          <circle cx="100" cy="100" r={R} stroke="var(--surface-2)" strokeWidth="16" fill="none" />
          <circle
            cx="100" cy="100" r={R} stroke={band.color} strokeWidth="16" fill="none"
            strokeLinecap="round" strokeDasharray={C} strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset .5s ease, stroke .3s ease' }}
          />
        </svg>
        <div className="ps-gauge-num">
          <b>{pct.toFixed(0)}%</b>
          <span>Riesgo estimado</span>
        </div>
      </div>
      <span className="ps-risk-pill" style={{ background: band.bg, color: band.color }}>{band.label}</span>
    </div>
  );
};

const Simulacion = () => {
  const [selectedDisease, setSelectedDisease] = useState('diabetes');
  const [config, setConfig] = useState(null);
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);

  const [currentResult, setCurrentResult] = useState(null);
  const [baseResult, setBaseResult] = useState(null);
  const [error, setError] = useState(null);
  const [baseCase, setBaseCase] = useState(null);

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

  // 2. Generar Caso Sintético
  const handleGenerateSynthetic = async () => {
    setLoading(true);
    try {
      const { data } = await getSyntheticCase(selectedDisease);
      const cleanData = {};
      const integers = ['age', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'hypertension', 'heart_disease'];
      const floats_1 = ['bmi', 'hba1c_level'];
      const floats_2 = ['diabetes_pedigree'];

      Object.keys(data).forEach(key => {
        let val = data[key];
        if (integers.includes(key)) {
          cleanData[key] = Math.round(val);
        } else if (floats_1.includes(key)) {
          cleanData[key] = parseFloat(Number(val).toFixed(1));
        } else if (floats_2.includes(key)) {
          cleanData[key] = parseFloat(Number(val).toFixed(2));
        } else {
          if (typeof val === 'number' && !key.includes('pedigree')) {
            cleanData[key] = Math.round(val);
          } else {
            cleanData[key] = val;
          }
        }
      });

      if (config.categoricals) {
        Object.keys(config.categoricals).forEach(cat => {
          const options = config.categoricals[cat];
          const activeOption = options.find(opt => data[`${cat}_${opt}`] === 1);
          cleanData[cat] = activeOption || options[0];
        });
      }

      setFormData(prev => ({ ...prev, ...cleanData }));
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

  // 3. Manejar cambios
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    if (currentResult && !baseResult) setCurrentResult(null);

    if (type === 'number') {
      if (value === '') {
        setFormData(prev => ({ ...prev, [name]: '' }));
        return;
      }
      setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const preparePayload = () => {
    const payload = { ...formData };
    if (payload.glucose !== undefined) payload.blood_glucose_level = payload.glucose;

    if (config.categoricals) {
      Object.keys(config.categoricals).forEach(catName => {
        const selectedValue = payload[catName];
        delete payload[catName];
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
    inputs: { ...inputs },
    result: { probability: apiData?.probability, prediction: apiData?.prediction },
    top_features: apiData?.top_features ?? null,
  });

  const saveSinglePredictionToHistory = (apiData) => {
    saveHistoryEntry({
      type: "single",
      id: makeId(),
      timestamp: Date.now(),
      disease: selectedDisease,
      inputs: { ...formData },
      result: { probability: apiData?.probability, prediction: apiData?.prediction },
      top_features: apiData?.top_features ?? null,
      notes: "",
    });
  };

  const saveComparisonToHistory = (newApiData) => {
    if (!baseCase) return;
    const newSnap = buildRunSnapshot(formData, newApiData);
    const basePct = (baseCase.result.probability ?? 0) * 100;
    const newPct = (newApiData.probability ?? 0) * 100;
    const diff = newPct - basePct;

    saveHistoryEntry({
      type: "comparison",
      id: makeId(),
      timestamp: Date.now(),
      disease: selectedDisease,
      base: { timestamp: baseCase.timestamp, inputs: { ...baseCase.inputs }, result: { ...baseCase.result } },
      scenarios: [{ timestamp: newSnap.timestamp, inputs: { ...newSnap.inputs }, result: { ...newSnap.result } }],
      delta: {
        probability_points: Number(diff.toFixed(1)),
        direction: diff < 0 ? "decrease" : diff > 0 ? "increase" : "same",
      },
      top_features_new: newApiData?.top_features ?? null,
      notes: "",
    });
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

      if (baseResult) saveComparisonToHistory(data);
      else saveSinglePredictionToHistory(data);
    } catch (err) {
      console.error(err);
      setError("Error al procesar la predicción. Revisa que todos los campos numéricos tengan valores.");
    } finally {
      setLoading(false);
    }
  };

  // 5. Comparación
  const handleSetBaseCase = () => {
    setBaseCase({
      timestamp: Date.now(),
      inputs: { ...formData },
      result: { probability: currentResult?.probability, prediction: currentResult?.prediction },
    });
    setBaseResult(currentResult);
    setCurrentResult(null);
    Swal.fire({
      icon: 'success',
      title: 'Escenario base fijado',
      text: 'Ahora modifica las variables (ej. baja el peso) y vuelve a calcular.',
    });
  };

  const handleResetComparison = () => {
    setBaseCase(null);
    setBaseResult(null);
    setCurrentResult(null);
  };

  // Campo numérico
  const renderNumberInput = (feat) => {
    const isCategoricalPart = config.categoricals && Object.keys(config.categoricals).some(cat => feat.startsWith(cat + "_"));
    if (isCategoricalPart) return null;
    if (feat === 'pregnancies' && formData['gender'] === 'Male') return null;

    const limits = CLINICAL_LIMITS[feat] || CLINICAL_LIMITS.default;
    return (
      <Form.Group className="ps-field" key={feat}>
        <Form.Label className="d-flex align-items-center justify-content-between">
          <span>{getLabel(feat)}<InfoIcon variableKey={feat} /></span>
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
        <Form.Text className="text-faint d-block text-end small">
          Recomendado: {limits.label}
        </Form.Text>
      </Form.Group>
    );
  };

  // Gráfico comparativo
  const renderComparisonChart = () => {
    if (!baseResult || !currentResult) return null;
    const data = [{
      name: 'Probabilidad de Riesgo',
      Base: (baseResult.probability * 100).toFixed(1),
      Nuevo: (currentResult.probability * 100).toFixed(1),
    }];
    const diff = (currentResult.probability * 100) - (baseResult.probability * 100);
    const isImprovement = diff < 0;

    return (
      <div className="mt-4">
        <hr style={{ borderColor: 'var(--border)' }} />
        <h5 className="mb-3"><BarChartSteps className="me-2" />Comparativa de escenarios</h5>
        <Alert variant={isImprovement ? 'success' : 'warning'}>
          <strong>Conclusión: </strong>
          {isImprovement
            ? `Con estos cambios, tu riesgo estimado se reduciría un ${Math.abs(diff).toFixed(1)} %.`
            : `Estos cambios aumentarían tu riesgo estimado en un ${Math.abs(diff).toFixed(1)} %.`}
        </Alert>
        <div style={{ width: '100%', height: 230 }}>
          <ResponsiveContainer>
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis type="number" domain={[0, 100]} unit="%" stroke="var(--text-faint)" />
              <YAxis type="category" dataKey="name" hide />
              <RechartsTooltip />
              <Legend />
              <Bar dataKey="Base" fill="#6f8a90" name="Escenario inicial" barSize={28} radius={[0, 6, 6, 0]} />
              <Bar dataKey="Nuevo" fill={isImprovement ? '#4fae8c' : '#c8736a'} name="Escenario simulado" barSize={28} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // Barras SHAP (estilo Pulso Sereno)
  const renderShap = () => {
    const feats = currentResult?.top_features || [];
    if (feats.length === 0) return null;
    const maxAbs = Math.max(...feats.map(f => Math.abs(Number(f.shap))), 1e-6);

    return (
      <div className="mt-4 text-start">
        <div className="text-faint mb-2" style={{ fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
          Factores que más influyeron (SHAP)
        </div>
        {feats.map((f, idx) => {
          const shapVal = Number(f.shap);
          const w = Math.max(8, (Math.abs(shapVal) / maxAbs) * 100);
          const pos = shapVal >= 0;
          return (
            <div className="ps-shap-row" key={idx}>
              <span className="lbl">{labelES(f.feature)}</span>
              <div className="ps-shap-track">
                <div className={`ps-shap-bar ${pos ? 'ps-shap-pos' : 'ps-shap-neg'}`} style={{ width: `${w}%` }}>
                  {pos ? '+' : '−'}{Math.abs(shapVal).toFixed(2)}
                </div>
              </div>
            </div>
          );
        })}
        <p className="small text-faint mt-2 mb-0">
          <ArrowUpShort className="text-danger" />Empuja el riesgo arriba ·
          <ArrowDownShort className="text-success" />lo reduce. SHAP explica la salida del modelo.
        </p>
      </div>
    );
  };

  // Capa clínica de referencia (ADA / ACC-AHA)
  const renderClinic = () => {
    if (!currentResult) return null;
    const flags = currentResult.clinical_flags || [];
    const note = currentResult.clinical_note;
    const sources = [...new Set(flags.map(f => f.source).filter(Boolean))];
    const refLine = sources.length
      ? `Ref: ${sources.join(' · ')} · Modelo ganador: ${prettyModel(currentResult.model)}`
      : `Modelo ganador: ${prettyModel(currentResult.model)}`;

    return (
      <div className="ps-clinic text-start">
        <h4><ClipboardData size={16} style={{ color: 'var(--accent)' }} /> Interpretación clínica de referencia</h4>
        <p>{note || 'Sin indicadores clínicos por encima de umbrales de referencia.'}</p>
        <div className="ref">{refLine}</div>
      </div>
    );
  };

  return (
    <Container className="py-5">
      <div className="ps-sec-head">
        <span className="ps-eyebrow">Simulador</span>
        <h2>Calcula y comprende tu riesgo</h2>
        <p>Ajusta tus datos y obtén una estimación con su explicación. Resultado educativo, no diagnóstico.</p>
      </div>

      {/* Selector de enfermedad */}
      <Nav variant="tabs" className="mb-4">
        {DISEASES.map(d => (
          <Nav.Item key={d}>
            <Nav.Link
              active={selectedDisease === d}
              onClick={() => setSelectedDisease(d)}
              className="text-capitalize px-4"
            >
              {getLabel(d)}
            </Nav.Link>
          </Nav.Item>
        ))}
      </Nav>

      {error && <Alert variant="danger">{error}</Alert>}

      {loading && !config ? (
        <div className="text-center py-5">
          <Spinner animation="border" variant="primary" />
          <p className="mt-2 text-faint">Cargando modelos de IA...</p>
        </div>
      ) : (
        config && (
          <div className="ps-sim-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1.15fr', gap: '24px', alignItems: 'start' }}>
            {/* IZQUIERDA: FORMULARIO */}
            <div className="ps-card">
              <div className="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                <h3 style={{ fontSize: '1.2rem', margin: 0 }}><PencilSquare className="me-2" />Tus datos</h3>
                <div className="d-flex gap-2">
                  {baseResult && (
                    <Button variant="outline-primary" size="sm" onClick={handleResetComparison}>
                      <ArrowCounterclockwise className="me-1" />Reiniciar
                    </Button>
                  )}
                  <OverlayTrigger placement="top" overlay={<Tooltip>Genera un caso ficticio con datos realistas</Tooltip>}>
                    <Button variant="outline-primary" size="sm" onClick={handleGenerateSynthetic} disabled={loading || !!baseResult}>
                      <Dice5Fill className="me-1" />Caso virtual
                    </Button>
                  </OverlayTrigger>
                </div>
              </div>

              {baseResult && (
                <Alert variant="info" className="py-2 mb-4">
                  <small><strong>Modo comparación:</strong> modifica los valores (ej. reduce el peso) y recalcula para ver el impacto.</small>
                </Alert>
              )}

              <Form onSubmit={handleSubmit}>
                {config.categoricals && Object.keys(config.categoricals).filter(cat => config.categoricals[cat].length > 0).map(cat => (
                  <Form.Group className="ps-field" key={cat}>
                    <Form.Label className="d-flex align-items-center justify-content-between">
                      <span>{getLabel(cat)}<InfoIcon variableKey={cat} /></span>
                    </Form.Label>
                    <Form.Select name={cat} value={formData[cat] || ''} onChange={handleChange}>
                      {config.categoricals[cat].map(opt => (
                        <option key={opt} value={opt}>{getLabel(opt)}</option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                ))}

                <Row>
                  {config.features.map(feat => {
                    const input = renderNumberInput(feat);
                    return input ? <Col sm={6} key={feat}>{input}</Col> : null;
                  })}
                </Row>

                <div className="d-grid mt-3">
                  <Button variant={baseResult ? 'success' : 'primary'} size="lg" type="submit" disabled={loading} className="fw-bold">
                    {loading
                      ? <Spinner as="span" animation="border" size="sm" />
                      : (baseResult ? <><Search className="me-2" />Comparar escenario</> : <><RocketTakeoffFill className="me-2" />Calcular riesgo</>)}
                  </Button>
                </div>
              </Form>
            </div>

            {/* DERECHA: RESULTADO */}
            <div className="ps-card">
              {!currentResult && !baseResult && (
                <div className="text-center text-faint py-5">
                  <HeartPulse size={56} style={{ color: 'var(--accent)', opacity: .55 }} />
                  <h4 className="mt-3" style={{ fontSize: '1.15rem' }}>Esperando datos…</h4>
                  <p className="small mb-0">Completa el formulario o usa «Caso virtual» para comenzar.</p>
                </div>
              )}

              {currentResult && (
                <div className="text-center">
                  <div className="d-flex justify-content-between align-items-center mb-2 flex-wrap gap-2">
                    <h3 style={{ fontSize: '1.2rem', margin: 0 }}>
                      {baseResult ? 'Nuevo escenario' : 'Resultado del análisis'}
                    </h3>
                    <span className="ps-tag d-inline-flex align-items-center" style={{ background: 'var(--surface-2)' }}>
                      <CpuFill className="me-1" />{prettyModel(currentResult.model)}
                    </span>
                  </div>

                  <Gauge pct={(currentResult.probability || 0) * 100} />
                  {renderShap()}
                  {renderClinic()}

                  {!baseResult && (
                    <div className="d-grid mt-4">
                      <Button variant="outline-primary" onClick={handleSetBaseCase}>
                        <ArrowLeftRight className="me-2" />Comparar con otro escenario
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {baseResult && !currentResult && (
                <div className="text-center text-faint py-5">
                  <ArrowRepeat size={48} style={{ color: 'var(--accent)', opacity: .6 }} />
                  <p className="small mt-3 mb-0">Escenario base fijado. Modifica los datos y recalcula.</p>
                </div>
              )}

              {renderComparisonChart()}
            </div>
          </div>
        )
      )}
    </Container>
  );
};

export default Simulacion;
