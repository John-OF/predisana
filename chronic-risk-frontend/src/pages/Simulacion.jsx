import { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Alert, Nav, Spinner, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { getConfig, predictRisk, getSyntheticCase, getWhatIf } from '../services/api';
import { getLabel } from '../utils/translations';
import AvisoSoporte from '../components/AvisoSoporte';
import Swal from 'sweetalert2';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend,
  ResponsiveContainer, AreaChart, Area, ReferenceLine, ReferenceDot, ReferenceArea,
} from 'recharts';
import {
  ArrowUpShort, ArrowDownShort, PencilSquare, ArrowCounterclockwise, Dice5Fill,
  RocketTakeoffFill, HeartPulse, ArrowLeftRight, Search, BarChartSteps, ClipboardData,
  ArrowRepeat, CpuFill, GraphUpArrow,
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

// Variables continuas que el panel "what-if" puede barrer, con su rango de barrido
// realista por enfermedad. Solo se ofrecen las que existan en config.features.
const WHATIF_FEATURES = {
  diabetes: [
    { feat: 'blood_glucose_level', min: 70, max: 300, step: 5 },
    { feat: 'age', min: 18, max: 90, step: 2 },
    { feat: 'bmi', min: 16, max: 45, step: 1 },
  ],
  hipertension: [
    { feat: 'age', min: 18, max: 90, step: 2 },
    { feat: 'bmi', min: 16, max: 45, step: 1 },
    { feat: 'waist_circumference', min: 60, max: 140, step: 5 },
    { feat: 'weight', min: 45, max: 140, step: 5 },
  ],
  cardiovascular: [
    { feat: 'ap_hi', min: 90, max: 200, step: 5 },
    { feat: 'ap_lo', min: 60, max: 130, step: 5 },
    { feat: 'bmi', min: 16, max: 45, step: 1 },
    { feat: 'age', min: 30, max: 90, step: 2 },
  ],
};

// --- AYUDA DE CADA CAMPO ---
// Rango recomendado que se muestra como ayuda y paso del input. El rango ACEPTADO
// (min/max) no vive aquí: lo sirve el backend en /config.ranges, que es el mismo que
// valida /predict. Antes había una copia a mano aquí que podía divergir del API.
const FIELD_HINTS = {
  age: { label: "18 - 100 años", step: 1 },
  glucose: { label: "70 - 200 mg/dL", step: 1 },
  blood_pressure: { label: "90 - 180 mmHg", step: 1 },
  bmi: { label: "18.5 - 40", step: 0.1 },
  hba1c_level: { label: "4 - 9 %", step: 0.1 },
  heart_disease: { label: "0 (No) - 1 (Sí)", step: 1 },
  hypertension: { label: "0 (No) - 1 (Sí)", step: 1 },
  diabetes: { label: "0 (No) - 1 (Sí)", step: 1 },
  high_cholesterol: { label: "0 (No) - 1 (Sí)", step: 1 },
  blood_glucose_level: { label: "70 - 200 mg/dL", step: 1 },
  weight: { label: "40 - 150 kg", step: 0.1 },
  waist_circumference: { label: "60 - 120 cm", step: 0.1 },
  ap_hi: { label: "90 - 180 mmHg", step: 1 },
  ap_lo: { label: "60 - 120 mmHg", step: 1 },
  cholesterol: { label: "1 (Normal) - 3 (Muy alto)", step: 1 },
  gluc: { label: "1 (Normal) - 3 (Muy alta)", step: 1 },
  smoke: { label: "0 (No) - 1 (Sí)", step: 1 },
  alco: { label: "0 (No) - 1 (Sí)", step: 1 },
  active: { label: "0 (No) - 1 (Sí)", step: 1 },
  default: { label: "Valor positivo", step: 1 }
};

// Copy del bloque de datos opcionales. Diabetes ofrece la glucosa (mejora el
// MODELO: activa la variante hibrida); hipertension ofrece la presion, que NO
// entra al modelo y solo se interpreta con la referencia ACC/AHA (AUD-1).
const OPTIONAL_COPY = {
  diabetes: {
    form: 'Sin glucosa, estimamos tu riesgo a partir de factores generales (edad, IMC, antecedentes), a modo de cribado. Si te la has medido, la estimación se vuelve mucho más precisa. La HbA1c, si la conoces, no cambia la estimación: se interpreta aparte con la referencia ADA.',
    result: 'Riesgo estimado a partir de factores generales, sin medir tu glucosa. Añádela arriba para una lectura mucho más precisa.',
  },
  hipertension: {
    form: 'El riesgo se estima con factores respondibles (edad, IMC, cintura, antecedentes), a modo de cribado. Si conoces tu presión, no cambia la estimación: se interpreta aparte con la referencia ACC/AHA.',
    result: 'Estimación de cribado a partir de factores generales. Si conoces tu presión, añádela arriba y la interpretamos con la referencia clínica.',
  },
  default: {
    form: 'Datos opcionales: si los conoces, afinan la lectura de tu resultado.',
    result: 'Estimación a partir de los datos que completaste.',
  },
};

// La clave `diabetes` ya existe en LABELS_ES como NOMBRE de enfermedad (la
// pestaña "Diabetes Tipo 2"). Como feature de hipertension significa otra cosa
// —"¿te lo han diagnosticado?"— asi que el formulario la reetiqueta aqui.
const FIELD_LABEL_OVERRIDES = {
  diabetes: 'Diabetes Previa',
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
  diabetes: "Indica si un médico te ha dicho alguna vez que tienes diabetes.",
  high_cholesterol: "Indica si un médico te ha dicho alguna vez que tienes el colesterol alto.",
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
  diabetes: "Diabetes Previa",
  high_cholesterol: "Colesterol Alto",
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

  // Panel "what-if": barrido de una variable manteniendo el resto fijo.
  const [whatIfFeat, setWhatIfFeat] = useState('');
  const [whatIf, setWhatIf] = useState(null); // { feature, curve, current }
  const [whatIfLoading, setWhatIfLoading] = useState(false);

  const resetWhatIf = () => { setWhatIfFeat(''); setWhatIf(null); };

  // 1. Cargar configuración
  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      setError(null);
      setCurrentResult(null);
      setBaseResult(null);
      setFormData({});
      resetWhatIf();
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
      // Sintético CTGAN en todas las enfermedades. Diabetes servía una ficha REAL de
      // NHANES (su sintético era del esquema viejo), presentada como "caso ficticio";
      // el sintético ya se regeneró al esquema NHANES y esa excepción sobraba.
      const { data } = await getSyntheticCase(selectedDisease);
      const cleanData = {};
      const integers = ['age', 'pregnancies', 'glucose', 'blood_glucose_level', 'blood_pressure', 'skin_thickness', 'insulin', 'hypertension', 'heart_disease'];
      const floats_1 = ['bmi', 'hba1c_level'];
      const floats_2 = ['diabetes_pedigree'];

      Object.keys(data).forEach(key => {
        let val = data[key];
        // Un dato ausente (null en la ficha) se deja vacío: Math.round(null) es 0 y el
        // formulario mostraba una glucosa de 0.
        if (val === null || val === undefined) return;
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
    if (currentResult && !baseResult) { setCurrentResult(null); resetWhatIf(); }

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
    // Features opcionales vacías (p.ej. glucosa que el usuario no aporta) NO se
    // envían: así el backend rutea al modelo self-report.
    Object.keys(payload).forEach(k => {
      if (payload[k] === '' || payload[k] === undefined || payload[k] === null) delete payload[k];
    });
    return payload;
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
      resetWhatIf();
    } catch (err) {
      console.error(err);
      if (err?.response?.status === 429) {
        // AUD-4: el API limita las peticiones por IP para que nadie inunde la demo.
        setError("Estás enviando demasiadas simulaciones seguidas. Espera un momento y vuelve a intentarlo.");
      } else {
        setError("Error al procesar la predicción. Revisa que todos los campos numéricos tengan valores.");
      }
    } finally {
      setLoading(false);
    }
  };

  // 5. Comparación
  const handleSetBaseCase = () => {
    setBaseResult(currentResult);
    setCurrentResult(null);
    Swal.fire({
      icon: 'success',
      title: 'Escenario base fijado',
      text: 'Ahora modifica las variables (ej. baja el peso) y vuelve a calcular.',
    });
  };

  const handleResetComparison = () => {
    setBaseResult(null);
    setCurrentResult(null);
  };

  // Campo numérico. `optional=true` no exige el campo (features tipo glucosa que
  // el usuario puede aportar o no; activan el modelo mejorado si las rellena).
  const renderNumberInput = (feat, optional = false) => {
    const isCategoricalPart = config.categoricals && Object.keys(config.categoricals).some(cat => feat.startsWith(cat + "_"));
    if (isCategoricalPart) return null;
    const hint = FIELD_HINTS[feat] || FIELD_HINTS.default;
    const [min, max] = config.ranges?.[feat] || [];
    // Opcional de la capa clínica (presión, HbA1c): se interpreta aparte y NO mueve el
    // número del modelo. La glucosa de diabetes, en cambio, sí lo afina.
    const soloClinico = optional && config.clinical_inputs?.includes(feat);
    return (
      <Form.Group className="ps-field" key={feat}>
        <Form.Label className="d-flex align-items-center justify-content-between">
          <span>
            {FIELD_LABEL_OVERRIDES[feat] || getLabel(feat)}<InfoIcon variableKey={feat} />
            {optional && <span className="ps-tag ms-2" style={{ background: 'var(--surface-2)', fontSize: '.68rem' }}>opcional</span>}
          </span>
        </Form.Label>
        <Form.Control
          type="number"
          name={feat}
          value={formData[feat] !== undefined ? formData[feat] : ''}
          onChange={handleChange}
          min={min}
          max={max}
          step={hint.step || "any"}
          placeholder={optional ? 'Déjalo vacío si no la conoces' : (min != null ? `Rango: ${min} - ${max}` : '')}
          required={!optional}
        />
        <Form.Text className="text-faint d-block text-end small">
          {optional
            ? (soloClinico ? 'No cambia la estimación: se interpreta aparte' : 'Si te la has medido, afina la estimación')
            : `Recomendado: ${hint.label}`}
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

  // Ejecuta el barrido what-if de una variable manteniendo el resto del caso fijo.
  const runWhatIf = async (feat) => {
    const spec = (WHATIF_FEATURES[selectedDisease] || []).find(s => s.feat === feat);
    if (!spec) { setWhatIf(null); return; }
    setWhatIfLoading(true);
    try {
      const base = preparePayload();
      const steps = Math.min(100, Math.max(2, Math.round((spec.max - spec.min) / spec.step) + 1));
      const { data } = await getWhatIf(selectedDisease, {
        base, feature: feat, min: spec.min, max: spec.max, steps,
      });
      const curve = (data.curve || []).map(p => ({ value: p.value, pct: +(p.probability * 100).toFixed(1) }));
      const currentVal = Number(formData[feat]);
      // Riesgo interpolado en el valor actual del usuario (para el punto marcado).
      let currentPct = null;
      if (Number.isFinite(currentVal) && curve.length) {
        const nearest = curve.reduce((a, b) =>
          Math.abs(b.value - currentVal) < Math.abs(a.value - currentVal) ? b : a);
        currentPct = nearest.pct;
      }
      setWhatIf({
        feature: feat, curve, current: Number.isFinite(currentVal) ? currentVal : null, currentPct,
        // AUD-16: hasta donde llegan los datos reales de entrenamiento.
        supported: data.supported_range || null,
        topcoded: data.topcoded_at ?? null,
      });
    } catch (err) {
      console.error(err);
      setWhatIf(null);
    } finally {
      setWhatIfLoading(false);
    }
  };

  // Panel "what-if": curva de riesgo al variar una sola variable.
  const renderWhatIf = () => {
    if (!currentResult || baseResult) return null;
    // Se puede barrer tanto features del modelo como las opcionales (p.ej. glucosa),
    // aunque el usuario no las haya rellenado: la curva muestra su efecto potencial.
    const available = [...(config.features || []), ...(config.optional_features || [])];
    const specs = (WHATIF_FEATURES[selectedDisease] || []).filter(s => available.includes(s.feat));
    if (!specs.length) return null;

    return (
      <div className="mt-4 text-start">
        <hr style={{ borderColor: 'var(--border)' }} />
        <h5 className="mb-2"><GraphUpArrow className="me-2" style={{ color: 'var(--accent)' }} />¿Y si cambiara una variable?</h5>
        <p className="small text-faint mb-3">
          Manteniendo el resto de tus datos igual, observa cómo se movería tu riesgo estimado
          al variar una sola variable. La curva usa la probabilidad calibrada del modelo.
        </p>
        <Form.Select
          value={whatIfFeat}
          onChange={(e) => { setWhatIfFeat(e.target.value); runWhatIf(e.target.value); }}
          className="mb-3"
        >
          <option value="">Elige una variable para explorar…</option>
          {specs.map(s => <option key={s.feat} value={s.feat}>{labelES(s.feat)}</option>)}
        </Form.Select>

        {whatIfLoading && (
          <div className="text-center py-4"><Spinner animation="border" size="sm" variant="primary" /></div>
        )}

        {whatIf && !whatIfLoading && (
          <>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer>
              <AreaChart data={whatIf.curve} margin={{ top: 6, right: 12, bottom: 22, left: 0 }}>
                <defs>
                  <linearGradient id="wiFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent, #2f9e8f)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--accent, #2f9e8f)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" dataKey="value" domain={['dataMin', 'dataMax']}
                  stroke="var(--text-faint)" fontSize={12}
                  label={{ value: labelES(whatIf.feature), position: 'insideBottom', offset: -12, fontSize: 12 }} />
                <YAxis domain={[0, 100]} unit="%" stroke="var(--text-faint)" fontSize={12} />
                <RechartsTooltip
                  formatter={(v) => [`${v}%`, 'Riesgo']}
                  labelFormatter={(v) => `${labelES(whatIf.feature)}: ${v}`} />
                <Area type="monotone" dataKey="pct" stroke="var(--accent, #2f9e8f)" strokeWidth={2}
                  fill="url(#wiFill)" />
                {/* AUD-16: el barrido puede pasarse del rango entrenado (edad hasta 100
                    con un modelo que vio hasta 65) y la curva no lo delata: se aplana
                    porque no hay datos, no porque el riesgo deje de subir. */}
                {whatIf.supported && whatIf.curve.length > 0 && (() => {
                  const [lo, hi] = whatIf.supported;
                  const x0 = whatIf.curve[0].value;
                  const x1 = whatIf.curve[whatIf.curve.length - 1].value;
                  return (
                    <>
                      {x0 < lo && <ReferenceArea x1={x0} x2={lo} fill="var(--text-faint)" fillOpacity={0.12} />}
                      {x1 > hi && <ReferenceArea x1={hi} x2={x1} fill="var(--text-faint)" fillOpacity={0.12} />}
                    </>
                  );
                })()}
                {whatIf.current != null && (
                  <ReferenceLine x={whatIf.current} stroke="var(--text-faint)" strokeDasharray="4 4"
                    label={{ value: 'tú', position: 'top', fontSize: 11, fill: 'var(--text-faint)' }} />
                )}
                {whatIf.current != null && whatIf.currentPct != null && (
                  <ReferenceDot x={whatIf.current} y={whatIf.currentPct} r={5}
                    fill="var(--accent, #2f9e8f)" stroke="#fff" strokeWidth={2} />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          {/* Fuera del div de altura fija del grafico: dentro se solapaba con el
              boton de comparar y el aviso se leia a medias. */}
          {whatIf.supported && (
            <p className="text-secondary small mt-2 mb-0">
              Zona sombreada: fuera de los datos de entrenamiento
              ({labelES(whatIf.feature).toLowerCase()} de {whatIf.supported[0]} a {whatIf.supported[1]}).
              {whatIf.topcoded != null
                ? ` En estos datos todo el que pasa de ${whatIf.topcoded} figura como ${whatIf.topcoded}: el modelo sí vio casos así, pero agrupados, y por encima la curva prolonga la tendencia.`
                : ' Ahí la curva es una extrapolación.'}
            </p>
          )}
          </>
        )}
      </div>
    );
  };

  const optionalCopy = OPTIONAL_COPY[selectedDisease] || OPTIONAL_COPY.default;
  const optionalNote = optionalCopy.form;
  const resultNote = optionalCopy.result;

  // Aviso de campos que el backend no recibió y asumió como 0. Antes la respuesta
  // traía `missing_filled_as_zero` y la UI lo ignoraba: el usuario veía un riesgo
  // calculado con un IMC de 0 sin enterarse.
  const renderMissing = () => {
    const faltantes = currentResult?.missing_filled_as_zero || [];
    if (!faltantes.length) return null;
    return (
      <Alert variant="warning" className="py-2 mt-3 text-start">
        <small>
          <strong>Sin dato:</strong> {faltantes.map(f => getLabel(f)).join(', ')}.
          Se calculó asumiendo 0, así que la estimación es menos fiable. Complétalos y recalcula.
        </small>
      </Alert>
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

                {config.optional_features && config.optional_features.length > 0 && (
                  <div className="mt-2 mb-1">
                    <div className="text-faint mb-2" style={{ fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
                      Datos opcionales — afinan la estimación
                    </div>
                    <p className="small text-soft mb-2">{optionalNote}</p>
                    <Row>
                      {config.optional_features.map(feat => {
                        const input = renderNumberInput(feat, true);
                        return input ? <Col sm={6} key={feat}>{input}</Col> : null;
                      })}
                    </Row>
                  </div>
                )}

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
                  {currentResult.used_glucose ? (
                    <div className="ps-tag mt-2 d-inline-flex align-items-center" style={{ background: 'var(--halo)', color: 'var(--accent)' }}>
                      <Search className="me-1" size={13} />Estimación mejorada con tu glucosa
                    </div>
                  ) : (config.optional_features?.length > 0 && (
                    <p className="small text-faint mt-2 mb-0">{resultNote}</p>
                  ))}
                  {renderMissing()}
                  {/* AUD-16: un modelo no avisa de que está extrapolando; el backend lo
                      calcula aparte (cardiovascular se entrenó con 29-65 años y aquí se
                      puede pedir 90). */}
                  <AvisoSoporte avisos={currentResult.support_warnings} />
                  {renderShap()}
                  {renderClinic()}
                  {renderWhatIf()}

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
