import { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Alert, Nav, Spinner, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { getConfig, predictRisk, getSyntheticCase, getWhatIf, MODOS, MODO_POR_DEFECTO } from '../services/api';
import { getLabel } from '../utils/translations';
import { riskBand, bandNote } from '../utils/riskBand';
import { FIELD_HINTS } from '../utils/fieldHints';
import AvisoSoporte from '../components/AvisoSoporte';
import Swal from 'sweetalert2';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend,
  ResponsiveContainer, AreaChart, Area, ReferenceLine, ReferenceDot, ReferenceArea,
} from 'recharts';
import {
  ArrowUpShort, ArrowDownShort, PencilSquare, ArrowCounterclockwise, Dice5Fill,
  RocketTakeoffFill, HeartPulse, ArrowLeftRight, Search, BarChartSteps, ClipboardData,
  ArrowRepeat, CpuFill, GraphUpArrow, PersonFill, Clipboard2PulseFill,
} from 'react-bootstrap-icons';

const DISEASES = ['diabetes', 'hipertension', 'cardiovascular', 'renal'];

const MODEL_LABELS = {
  logistic_regression: 'Regresión Logística',
  logreg: 'Regresión Logística',
  random_forest: 'Random Forest',
  lightgbm: 'LightGBM',
  xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');

// Dos modelos por enfermedad (v2). El simplificado se entrena con el peso y la talla que
// la gente declara (que es lo que va a escribir aquí); el completo, con lo medido.
const MODE_COPY = {
  simplificado: {
    icon: PersonFill,
    desc: 'Lo que cualquiera sabe de sí mismo: edad, sexo, peso, talla, tabaco y diagnósticos previos.',
    optional: (guias) => `Si los conoces, no cambian la estimación: se interpretan aparte con las guías${guias ? ` (${guias})` : ''}.`,
  },
  completo: {
    icon: Clipboard2PulseFill,
    desc: 'Para personal sanitario: añade medidas y análisis (IMC medido, cintura, presión, colesterol, función renal).',
    optional: (guias) => `No cambian la estimación: o definen la enfermedad (lo dice la guía, sin necesidad de un modelo) o el modelo no encuentra en ellos señal. Se interpretan aparte con las guías${guias ? ` (${guias})` : ''}.`,
  },
};

// Guía que interpreta cada dato opcional: el texto del bloque cita solo las que tocan.
const GUIAS = {
  blood_glucose_level: 'ADA', hba1c_level: 'ADA', ap_hi: 'ACC/AHA', ap_lo: 'ACC/AHA',
  egfr: 'KDIGO', albumin_creatinine_ratio: 'KDIGO',
};

// Rango de barrido del what-if por variable. Solo se ofrecen las que el modelo de ese
// modo usa de verdad (el backend rechaza el resto: darían una recta).
const WHATIF_RANGOS = {
  age: { min: 18, max: 90, step: 2 },
  weight: { min: 45, max: 140, step: 5 },
  bmi: { min: 16, max: 45, step: 1 },
  waist_circumference: { min: 60, max: 140, step: 5 },
  ap_hi: { min: 90, max: 200, step: 5 },
  ap_lo: { min: 50, max: 120, step: 5 },
  total_cholesterol: { min: 120, max: 320, step: 10 },
  hdl_cholesterol: { min: 25, max: 100, step: 5 },
  hba1c_level: { min: 4, max: 10, step: 0.25 },
  egfr: { min: 20, max: 130, step: 5 },
  albumin_creatinine_ratio: { min: 0, max: 300, step: 10 },
};

// La clave `diabetes` ya existe en LABELS_ES como NOMBRE de enfermedad (la
// pestaña "Diabetes Tipo 2"). Como variable de los otros modelos significa otra cosa
// —"¿te lo han diagnosticado?"— asi que el formulario la reetiqueta aqui.
const FIELD_LABEL_OVERRIDES = {
  diabetes: 'Diabetes Previa',
};

const VARIABLE_DESCRIPTIONS = {
  age: "La edad influye directamente en el riesgo acumulado. A mayor edad, mayor vigilancia requerida.",
  gender: "Factor biológico que puede predisponer a ciertas condiciones hormonales o cardiovasculares.",
  smoking_history: "Fumar daña las arterias y el corazón. Es el factor de riesgo modificable más crítico.",
  weight: "Tu peso en kilogramos. Con tu talla, calculamos tu IMC.",
  height: "Tu talla en centímetros. Con tu peso, calculamos tu IMC.",
  bmi: "Índice de Masa Corporal medido. Un valor superior a 25 indica sobrepeso; superior a 30, obesidad.",
  heart_disease: "Si un médico te ha dicho alguna vez que tuviste un infarto, angina, insuficiencia cardiaca, enfermedad coronaria o un ictus.",
  hypertension: "Si un médico te ha dicho alguna vez que tienes la presión alta.",
  diabetes: "Si un médico te ha dicho alguna vez que tienes diabetes.",
  high_cholesterol: "Si un médico te ha dicho alguna vez que tienes el colesterol alto.",
  waist_circumference: "Contorno de cintura en cm. Refleja la grasa abdominal, clave en el riesgo metabólico.",
  ap_hi: "Presión sistólica (la 'alta'): el primer número al medir la presión.",
  ap_lo: "Presión diastólica (la 'baja'): el segundo número al medir la presión.",
  total_cholesterol: "Colesterol total en sangre (mg/dL).",
  hdl_cholesterol: "Colesterol HDL, el 'bueno': cuanto más alto, mejor.",
  hba1c_level: "Hemoglobina Glicosilada. Es tu 'promedio' de azúcar de los últimos 3 meses. Ideal: menos de 5.7%.",
  blood_glucose_level: "Glucosa en sangre tras 8 horas sin comer. Lo normal: 70 a 99 mg/dL.",
  egfr: "Filtrado glomerular estimado: cuánto filtran los riñones (mL/min/1,73 m²). Por debajo de 60, enfermedad renal.",
  albumin_creatinine_ratio: "Albúmina en orina frente a creatinina (mg/g). Por encima de 30, daño renal.",
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
  bmi: "Índice de Masa Corporal (IMC)",
  age: "Edad",
  heart_disease: "Enfermedad Cardiovascular Previa",
  hypertension: "Hipertensión Previa",
  diabetes: "Diabetes Previa",
  high_cholesterol: "Colesterol Alto",
  waist_circumference: "Cintura (cm)",
  ap_hi: "Presión Sistólica",
  ap_lo: "Presión Diastólica",
  total_cholesterol: "Colesterol Total",
  hdl_cholesterol: "Colesterol HDL",
  hba1c_level: "HbA1c",
  egfr: "Filtrado Glomerular",
  albumin_creatinine_ratio: "Albúmina en Orina",
  smoking_history_current: "Tabaquismo: Actual",
  smoking_history_former: "Tabaquismo: Exfumador",
  gender_Male: "Sexo: Masculino",
  gender_Female: "Sexo: Femenino",
};

const labelES = (feat) => {
  const key = String(feat || "").trim().replace(/\s+/g, "_");
  return SHAP_LABELS_ES[key] || FIELD_LABEL_OVERRIDES[key] || getLabel(feat) || feat;
};

// Medidor circular. `band` viene de riskBand(resultado): la decide el backend.
const Gauge = ({ pct, band }) => {
  const R = 84;
  const C = 2 * Math.PI * R; // ≈ 528
  const offset = C - (C * pct) / 100;
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
  const [selectedMode, setSelectedMode] = useState(MODO_POR_DEFECTO);
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

  // 1. Cargar configuración del modelo (enfermedad + modo). Lo que el usuario ya
  // escribió y sigue teniendo sentido (edad, sexo, diagnósticos) se conserva.
  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      setError(null);
      setCurrentResult(null);
      setBaseResult(null);
      resetWhatIf();
      try {
        const { data } = await getConfig(selectedDisease, selectedMode);
        setConfig(data);
        const campos = [...(data.inputs || []), ...(data.optional_features || [])];
        setFormData(prev => {
          const siguiente = {};
          campos.forEach(k => { if (prev[k] !== undefined) siguiente[k] = prev[k]; });
          Object.entries(data.categoricals || {}).forEach(([cat, opciones]) => {
            if (!opciones.includes(siguiente[cat])) {
              siguiente[cat] = opciones.includes('never') ? 'never' : opciones[0];
            }
          });
          return siguiente;
        });
      } catch (err) {
        console.error(err);
        setError("Error cargando la configuración del modelo.");
      } finally {
        setLoading(false);
      }
    };
    fetchConfig();
  }, [selectedDisease, selectedMode]);

  const camposDelModo = () => new Set([...(config?.inputs || []), ...(config?.optional_features || [])]);

  // 2. Generar Caso Sintético (CTGAN): trae todas las variables de la enfermedad, y se
  // cargan las que pide este modo.
  const handleGenerateSynthetic = async () => {
    setLoading(true);
    try {
      const { data } = await getSyntheticCase(selectedDisease);
      const cleanData = {};
      camposDelModo().forEach(k => {
        const opciones = config.categoricals?.[k];
        if (opciones) {
          const activa = opciones.find(opt => data[`${k}_${opt}`] === 1);
          if (activa) cleanData[k] = activa;
          return;
        }
        const val = data[k];
        // Un dato ausente (null en la ficha) se deja vacío: Math.round(null) es 0 y el
        // formulario mostraba una glucosa de 0.
        if (val === null || val === undefined) return;
        cleanData[k] = k === 'age' ? Math.round(val) : Number(Number(val).toFixed(1));
      });

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

  // Solo los campos de este modo: un peso que quedó escrito en el simplificado no debe
  // viajar al completo, que usa el IMC medido.
  const preparePayload = () => {
    const payload = {};
    const campos = camposDelModo();
    Object.entries(formData).forEach(([k, v]) => {
      if (!campos.has(k)) return;
      const opciones = config.categoricals?.[k];
      if (opciones) {
        // Un grupo (sexo, tabaco) viaja como una columna por opción, todas: la API exige
        // una y solo una aunque el modelo no las lea todas.
        opciones.forEach(opt => { payload[`${k}_${opt}`] = (opt === v) ? 1 : 0; });
      } else if (v !== '' && v !== undefined && v !== null) {
        payload[k] = v;
      }
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
      const { data } = await predictRisk(selectedDisease, payload, selectedMode);
      setCurrentResult(data);
      resetWhatIf();
    } catch (err) {
      console.error(err);
      if (err?.response?.status === 429) {
        // AUD-4: el API limita las peticiones por IP para que nadie inunde la demo.
        setError("Estás enviando demasiadas simulaciones seguidas. Espera un momento y vuelve a intentarlo.");
      } else if (err?.response?.status === 400 && err.response.data?.error) {
        setError(`Revisa los datos: ${err.response.data.error}.`);
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
      text: 'Ahora modifica las variables (ej. baja el IMC) y vuelve a calcular.',
    });
  };

  const handleResetComparison = () => {
    setBaseResult(null);
    setCurrentResult(null);
  };

  // Campo numérico. `aparte=true`: no entra al modelo (lo interpreta la guía, o el modelo
  // no lo usa), así que no es obligatorio.
  const renderNumberInput = (feat, aparte = false) => {
    const hint = FIELD_HINTS[feat] || FIELD_HINTS.default;
    const [min, max] = config.ranges?.[feat] || [];
    return (
      <Form.Group className="ps-field" key={feat}>
        <Form.Label className="d-flex align-items-center justify-content-between">
          <span>
            {FIELD_LABEL_OVERRIDES[feat] || getLabel(feat)}<InfoIcon variableKey={feat} />
            {aparte && <span className="ps-tag ms-2" style={{ background: 'var(--surface-2)', fontSize: '.68rem' }}>opcional</span>}
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
          placeholder={aparte ? 'Déjalo vacío si no lo conoces' : (min != null ? `Rango: ${min} - ${max}` : '')}
          required={!aparte}
        />
        <Form.Text className="text-faint d-block text-end small">
          {/* Lo que no tiene guía (el colesterol alto de la renal) no "se interpreta": el
              modelo, simplemente, no le da peso. */}
          {aparte
            ? (GUIAS[feat] ? `No cambia la estimación: lo interpreta la guía (${GUIAS[feat]})` : 'No cambia la estimación: el modelo no le da peso')
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

  // Variables que el what-if puede barrer en este modo: las que el modelo usa de verdad.
  const whatIfOpciones = () => (config?.inputs || [])
    .filter(f => WHATIF_RANGOS[f] && !(config.not_used || []).includes(f));

  // Ejecuta el barrido what-if de una variable manteniendo el resto del caso fijo.
  const runWhatIf = async (feat) => {
    const spec = WHATIF_RANGOS[feat];
    if (!spec) { setWhatIf(null); return; }
    setWhatIfLoading(true);
    try {
      const base = preparePayload();
      const steps = Math.min(100, Math.max(2, Math.round((spec.max - spec.min) / spec.step) + 1));
      const { data } = await getWhatIf(selectedDisease, {
        base, feature: feat, min: spec.min, max: spec.max, steps, mode: selectedMode,
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
        // En el simplificado, barrer el peso mueve el IMC a la talla del caso base.
        coupled: data.coupled || null,
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
    const opciones = whatIfOpciones();
    if (!opciones.length) return null;

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
          {opciones.map(f => <option key={f} value={f}>{labelES(f)}</option>)}
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
                {/* AUD-16: el barrido puede pasarse del rango entrenado y la curva no lo
                    delata: se aplana porque no hay datos, no porque el riesgo deje de subir. */}
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
          {/* Sin esto la curva de peso contestaba otra pregunta: con el IMC quieto, mas
              peso es mas estatura. El backend mueve los dos a la vez y lo declara en
              `coupled`. */}
          {whatIf.coupled && (
            <p className="text-secondary small mt-2 mb-0">
              Peso e IMC se mueven juntos: la curva mantiene tu talla
              (≈ {whatIf.coupled.height_m.toFixed(2).replace('.', ',')} m), porque cambiar
              uno sin el otro sería cambiar de estatura.
            </p>
          )}
          </>
        )}
      </div>
    );
  };

  // Capa clínica de referencia (ADA / ACC-AHA)
  const renderClinic = () => {
    if (!currentResult) return null;
    const flags = currentResult.clinical_flags || [];
    const note = currentResult.clinical_note;
    // "modelo" no es una guía: es el aviso de las variables que el modelo no usa.
    const sources = [...new Set(flags.map(f => f.source).filter(s => s && s !== 'modelo'))];
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

  const modeCopy = MODE_COPY[selectedMode];
  const notUsed = config?.not_used || [];
  const principales = (config?.inputs || []).filter(f => !config.categoricals?.[f] && !notUsed.includes(f));
  const aparte = [...notUsed.filter(f => (config?.inputs || []).includes(f)), ...(config?.optional_features || [])];
  const guias = [...new Set(aparte.map(f => GUIAS[f]).filter(Boolean))].join(', ');

  return (
    <Container className="py-5">
      <div className="ps-sec-head">
        <span className="ps-eyebrow">Simulador</span>
        <h2>Calcula y comprende tu riesgo</h2>
        <p>Ajusta tus datos y obtén una estimación con su explicación. Resultado educativo, no diagnóstico.</p>
      </div>

      {/* Selector de enfermedad */}
      <Nav variant="tabs" className="mb-3">
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

      {/* Selector de modo: dos modelos por enfermedad */}
      <div className="d-flex flex-wrap align-items-center gap-3 mb-4">
        <Nav variant="pills">
          {MODOS.map(m => {
            const Icono = MODE_COPY[m].icon;
            return (
              <Nav.Item key={m}>
                <Nav.Link active={selectedMode === m} onClick={() => setSelectedMode(m)} className="px-3">
                  <Icono className="me-2" />{getLabel(m)}
                </Nav.Link>
              </Nav.Item>
            );
          })}
        </Nav>
        <span className="small text-soft">{modeCopy.desc}</span>
      </div>

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
                  <small><strong>Modo comparación:</strong> modifica los valores (ej. reduce el IMC) y recalcula para ver el impacto.</small>
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
                  {principales.map(feat => <Col sm={6} key={feat}>{renderNumberInput(feat)}</Col>)}
                </Row>

                {aparte.length > 0 && (
                  <div className="mt-2 mb-1">
                    <div className="text-faint mb-2" style={{ fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
                      {selectedMode === 'completo' ? 'Se interpretan con la guía' : 'Si los conoces'}
                    </div>
                    <p className="small text-soft mb-2">{modeCopy.optional(guias)}</p>
                    <Row>
                      {aparte.map(feat => <Col sm={6} key={feat}>{renderNumberInput(feat, true)}</Col>)}
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
                      <CpuFill className="me-1" />{prettyModel(currentResult.model)} · {getLabel(currentResult.mode)}
                    </span>
                  </div>

                  <Gauge pct={(currentResult.probability || 0) * 100} band={riskBand(currentResult)} />
                  {bandNote(currentResult.risk_bands) && (
                    <p className="small text-faint mt-2 mb-0">{bandNote(currentResult.risk_bands)}</p>
                  )}
                  {currentResult.mode === 'simplificado' && currentResult.bmi != null && (
                    <p className="small text-faint mt-1 mb-0">
                      IMC calculado con tu peso y tu talla: {Number(currentResult.bmi).toFixed(1).replace('.', ',')}
                    </p>
                  )}
                  {/* AUD-16: un modelo no avisa de que está extrapolando; el backend lo
                      calcula aparte. */}
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
