import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Container, Row, Col, Card, Button, Table, Badge, Accordion, Spinner, Tabs, Tab } from 'react-bootstrap';
import { getSyntheticCase, getMetrics, predictRisk, getSampleCase, getDistribution, getSyntheticQuality } from '../services/api';
import { getLabel } from '../utils/translations';
import { Droplet, HeartPulse, Heart, Eyedropper, Robot, Lightbulb, Magic, Stars, BarChartLineFill, ArrowRight, TrophyFill, CpuFill, ArrowUpShort, ArrowDownShort, ArrowRepeat, PatchQuestion, ClipboardCheck, ClipboardPulse, CodeSlash, Window, Github, PersonBadge } from 'react-bootstrap-icons';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Cada enfermedad tiene un esquema de datos distinto (diabetes no mide presión
// continua, cardiovascular usa colesterol/glucosa ordinales 1-3, hipertensión
// trae cintura/peso...). Por eso la interpretación clínica es por-enfermedad.
const DISEASE_TABS = [
    { key: 'diabetes', label: 'Diabetes', icon: <Droplet className="me-2" />, variant: 'primary' },
    { key: 'hipertension', label: 'Hipertensión', icon: <HeartPulse className="me-2" />, variant: 'danger' },
    { key: 'cardiovascular', label: 'Cardiovascular', icon: <Heart className="me-2" />, variant: 'info' },
];

const diseaseLabel = (key) => DISEASE_TABS.find((t) => t.key === key)?.label ?? key;

// Variables numéricas continuas por enfermedad para el histograma comparado.
const DIST_FEATURES = {
    diabetes: ['age', 'bmi', 'blood_glucose_level', 'hba1c_level'],
    hipertension: ['age', 'bmi', 'weight', 'waist_circumference'],
    cardiovascular: ['age', 'bmi', 'ap_hi', 'ap_lo'],
};

// Nombres legibles de los algoritmos del leaderboard (claves que emite el backend).
const MODEL_LABELS = {
    logreg: 'Reg. Logística',
    logistic_regression: 'Reg. Logística',
    random_forest: 'Random Forest',
    lightgbm: 'LightGBM',
    xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');

// Banda de riesgo serena (misma escala que el Simulador).
const riskBand = (pct) => {
    if (pct < 33) return { label: 'Riesgo bajo', color: 'var(--risk-low)', bg: 'var(--risk-low-bg)' };
    if (pct < 66) return { label: 'Riesgo moderado', color: 'var(--risk-mid)', bg: 'var(--risk-mid-bg)' };
    return { label: 'Riesgo alto', color: 'var(--risk-high)', bg: 'var(--risk-high-bg)' };
};

// Quita los campos meta (_source_type, _disease) y el target antes de predecir.
const stripMeta = (obj) => {
    const out = {};
    Object.keys(obj).forEach((k) => { if (!k.startsWith('_') && k !== 'target') out[k] = obj[k]; });
    return out;
};

// --- Helpers de interpretación (badges) ---
const imcBadge = (bmi) =>
    bmi >= 30 ? <Badge bg="danger">Obesidad</Badge> :
    bmi >= 25 ? <Badge bg="warning">Sobrepeso</Badge> :
    <Badge bg="success">Normal</Badge>;

const glucosaBadge = (g) =>
    g > 200 ? <Badge bg="danger">Diabetes</Badge> :
    g > 100 ? <Badge bg="warning">Riesgo</Badge> :
    <Badge bg="success">Normal</Badge>;

const presionBadge = (sys) =>
    sys >= 140 ? <Badge bg="danger">Hipertensión</Badge> :
    sys >= 130 ? <Badge bg="warning">Elevada</Badge> :
    <Badge bg="success">Normal</Badge>;

// cholesterol / gluc de cardiovascular son ordinales: 1=normal, 2=alto, 3=muy alto
const ordinalBadge = (v) =>
    Number(v) === 3 ? <Badge bg="danger">Muy alto</Badge> :
    Number(v) === 2 ? <Badge bg="warning">Elevado</Badge> :
    <Badge bg="success">Normal</Badge>;

const siNoBadge = (v, texto) =>
    Number(v) === 1 ? <Badge bg="danger">{texto}</Badge> : <span className="text-muted">—</span>;

const getGenderLabel = (data) => {
    if (data.gender_Male === undefined && data.gender_Female === undefined) return 'No especificado';
    if (Number(data.gender_Male) === 1) return 'Masculino';
    if (Number(data.gender_Female) === 1) return 'Femenino';
    return 'Otro';
};

// Devuelve las filas {label, valor, interp} a renderizar según el esquema real
// que /synthetic/<disease> entrega para cada enfermedad.
const buildRows = (disease, d) => {
    const edadGenero = { label: 'Edad / Género', valor: `${Math.floor(d.age)} años / ${getGenderLabel(d)}`, interp: <span className="text-muted">Demográfico</span> };
    const imc = { label: 'IMC (Masa Corporal)', valor: Number(d.bmi).toFixed(1), interp: imcBadge(d.bmi) };

    if (disease === 'diabetes') {
        return [
            edadGenero,
            imc,
            { label: 'Glucosa', valor: `${Math.round(d.blood_glucose_level)} mg/dL`, interp: glucosaBadge(d.blood_glucose_level) },
            { label: 'HbA1c', valor: `${Number(d.hba1c_level).toFixed(1)} %`, interp: d.hba1c_level >= 6.5 ? <Badge bg="danger">Diabetes</Badge> : d.hba1c_level >= 5.7 ? <Badge bg="warning">Prediabetes</Badge> : <Badge bg="success">Normal</Badge> },
            { label: 'Hipertensión (Dx)', valor: Number(d.hypertension) === 1 ? 'Sí' : 'No', interp: siNoBadge(d.hypertension, 'Diagnóstico Presente') },
            { label: 'Enfermedad Cardíaca', valor: Number(d.heart_disease) === 1 ? 'Sí' : 'No', interp: siNoBadge(d.heart_disease, 'Historial Presente') },
        ];
    }

    if (disease === 'hipertension') {
        return [
            edadGenero,
            imc,
            { label: 'Circunferencia de Cintura', valor: `${Math.round(d.waist_circumference)} cm`, interp: <span className="text-muted">Adiposidad central</span> },
            { label: 'Peso', valor: `${Math.round(d.weight)} kg`, interp: <span className="text-muted">Antropometría</span> },
            { label: 'Diabetes (Dx)', valor: Number(d.diabetes) === 1 ? 'Sí' : 'No', interp: siNoBadge(d.diabetes, 'Diagnóstico Presente') },
            { label: 'Colesterol Alto (Dx)', valor: Number(d.high_cholesterol) === 1 ? 'Sí' : 'No', interp: siNoBadge(d.high_cholesterol, 'Diagnóstico Presente') },
        ];
    }

    // cardiovascular
    return [
        edadGenero,
        imc,
        { label: 'Presión (Sistólica / Diastólica)', valor: `${Math.round(d.ap_hi)} / ${Math.round(d.ap_lo)} mmHg`, interp: presionBadge(d.ap_hi) },
        { label: 'Colesterol', valor: `Nivel ${Number(d.cholesterol)}`, interp: ordinalBadge(d.cholesterol) },
        { label: 'Glucosa', valor: `Nivel ${Number(d.gluc)}`, interp: ordinalBadge(d.gluc) },
        { label: 'Hábitos', valor: 'Tabaco / Alcohol / Actividad', interp: <span className="d-flex gap-1 justify-content-center flex-wrap">{siNoBadge(d.smoke, 'Fuma')}{siNoBadge(d.alco, 'Alcohol')}{Number(d.active) === 1 ? <Badge bg="success">Activo</Badge> : <Badge bg="secondary">Sedentario</Badge>}</span> },
    ];
};

const Proyecto = () => {
    const [syntheticData, setSyntheticData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [disease, setDisease] = useState('diabetes');
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);

    // Juego "¿cuál es real?": un duelo real vs sintético.
    const [gameDisease, setGameDisease] = useState('diabetes');
    const [duel, setDuel] = useState(null);       // { cards: [c0, c1], realIndex }
    const [guess, setGuess] = useState(null);     // índice elegido por el usuario
    const [loadingDuel, setLoadingDuel] = useState(false);

    // Distribuciones real vs sintético.
    const [distDisease, setDistDisease] = useState('diabetes');
    const [distFeature, setDistFeature] = useState('age');
    const [distData, setDistData] = useState(null);
    const [loadingDist, setLoadingDist] = useState(false);

    // Calidad del sintético (SDMetrics + correlaciones).
    const [qualDisease, setQualDisease] = useState('diabetes');
    const [qualData, setQualData] = useState(null);
    const [loadingQual, setLoadingQual] = useState(false);

    // Pestaña activa del laboratorio (controlada: difiere las cargas pesadas).
    const [labTab, setLabTab] = useState('generar');

    // Panorámica de selección de modelos: se leen los 3 leaderboards en vivo
    // (mismo endpoint que /metricas) para no hardcodear AUCs que se desincronicen.
    const [modelsByDisease, setModelsByDisease] = useState(null);
    const [loadingModels, setLoadingModels] = useState(true);

    useEffect(() => {
        const keys = DISEASE_TABS.map((t) => t.key);
        Promise.all(keys.map((k) => getMetrics(k).then((res) => [k, res.data])))
            .then((pairs) => setModelsByDisease(Object.fromEntries(pairs)))
            .catch((err) => {
                console.error('Error cargando métricas de modelos:', err);
                setModelsByDisease(null);
            })
            .finally(() => setLoadingModels(false));
    }, []);

    useEffect(() => {
        if (labTab !== 'dist') return;
        let cancel = false;
        setLoadingDist(true);
        getDistribution(distDisease, distFeature)
            .then((res) => { if (!cancel) setDistData(res.data); })
            .catch((err) => { console.error('Error cargando distribución:', err); if (!cancel) setDistData(null); })
            .finally(() => { if (!cancel) setLoadingDist(false); });
        return () => { cancel = true; };
    }, [labTab, distDisease, distFeature]);

    useEffect(() => {
        if (labTab !== 'calidad') return;
        let cancel = false;
        setLoadingQual(true);
        getSyntheticQuality(qualDisease)
            .then((res) => { if (!cancel) setQualData(res.data); })
            .catch((err) => { console.error('Error cargando calidad:', err); if (!cancel) setQualData(null); })
            .finally(() => { if (!cancel) setLoadingQual(false); });
        return () => { cancel = true; };
    }, [labTab, qualDisease]);

    const generateDemo = async (target) => {
        const dis = target || disease;
        setDisease(dis);
        setLoading(true);
        setSyntheticData(null);
        setPrediction(null);
        try {
            const { data } = await getSyntheticCase(dis);
            setSyntheticData({ ...data, _disease: dis });
        } catch (error) {
            console.error("Error generando caso:", error);
        } finally {
            setLoading(false);
        }
    };

    // Bucle: el paciente sintético se pasa por el modelo (cierra CTGAN → modelo → SHAP).
    const evaluateSynthetic = async () => {
        if (!syntheticData) return;
        setPredicting(true);
        try {
            const { data } = await predictRisk(syntheticData._disease, stripMeta(syntheticData));
            setPrediction(data);
        } catch (err) {
            console.error('Error evaluando paciente:', err);
        } finally {
            setPredicting(false);
        }
    };

    const renderPrediction = () => {
        if (!prediction) return null;
        const pct = (prediction.probability || 0) * 100;
        const band = riskBand(pct);
        const feats = prediction.top_features || [];
        const maxAbs = Math.max(...feats.map((f) => Math.abs(Number(f.shap))), 1e-6);
        return (
            <Card className="border-0 shadow-sm mt-3">
                <Card.Body>
                    <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 mb-3">
                        <div className="d-flex align-items-center gap-3">
                            <div>
                                <div className="text-faint" style={{ fontSize: '.74rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>Riesgo estimado</div>
                                <div style={{ fontFamily: 'var(--font-display)', fontSize: '2.1rem', lineHeight: 1.1, color: band.color }}>{pct.toFixed(0)}%</div>
                            </div>
                            <span className="ps-risk-pill" style={{ background: band.bg, color: band.color }}>{band.label}</span>
                        </div>
                        {prediction.model && <span className="ps-tag"><CpuFill className="me-1" />{prettyModel(prediction.model)}</span>}
                    </div>

                    {feats.length > 0 && (
                        <div className="text-start">
                            <div className="text-faint mb-2" style={{ fontSize: '.74rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
                                Factores que más influyeron (SHAP)
                            </div>
                            {feats.map((f, idx) => {
                                const shapVal = Number(f.shap);
                                const w = Math.max(8, (Math.abs(shapVal) / maxAbs) * 100);
                                const pos = shapVal >= 0;
                                return (
                                    <div className="ps-shap-row" key={idx}>
                                        <span className="lbl">{getLabel(f.feature) || f.feature}</span>
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
                    )}

                    {prediction.clinical_note && (
                        <div className="ps-clinic text-start mt-3">
                            <p className="mb-0">{prediction.clinical_note}</p>
                        </div>
                    )}
                </Card.Body>
            </Card>
        );
    };

    // --- Juego "¿cuál es real?" ---
    const newRound = async (target) => {
        const dis = target || gameDisease;
        setGameDisease(dis);
        setLoadingDuel(true);
        setDuel(null);
        setGuess(null);
        try {
            const [realRes, synthRes] = await Promise.all([
                getSampleCase(dis, 'real'),
                getSampleCase(dis, 'synthetic'),
            ]);
            const real = { ...realRes.data, _disease: dis };
            const fake = { ...synthRes.data, _disease: dis };
            const realIndex = Math.random() < 0.5 ? 0 : 1;
            setDuel({ cards: realIndex === 0 ? [real, fake] : [fake, real], realIndex });
        } catch (err) {
            console.error('Error en ronda real vs sintético:', err);
        } finally {
            setLoadingDuel(false);
        }
    };

    const renderDuelCard = (card, index) => {
        const revealed = guess !== null;
        const isReal = duel.realIndex === index;
        const chosen = guess === index;
        const borderClass = revealed && chosen ? (isReal ? 'border-success border-2' : 'border-danger border-2') : 'border-0';
        return (
            <Card className={`shadow-sm h-100 ${borderClass}`}>
                <Card.Header className="d-flex justify-content-between align-items-center">
                    <span className="fw-bold">Paciente {index === 0 ? 'A' : 'B'}</span>
                    {revealed
                        ? (isReal ? <Badge bg="success">REAL</Badge> : <Badge bg="warning" text="dark">SINTÉTICO</Badge>)
                        : <Badge bg="secondary">¿?</Badge>}
                </Card.Header>
                <Table striped responsive className="mb-0 align-middle small">
                    <tbody>
                        {buildRows(card._disease, card).map((row, i) => (
                            <tr key={i}>
                                <td className="text-start ps-3 fw-bold">{row.label}</td>
                                <td className="text-end pe-3">{row.valor}</td>
                            </tr>
                        ))}
                    </tbody>
                </Table>
                {!revealed && (
                    <Card.Footer className="text-center bg-white">
                        <Button size="sm" variant="outline-primary" onClick={() => setGuess(index)}>
                            Creo que es la REAL
                        </Button>
                    </Card.Footer>
                )}
            </Card>
        );
    };

    const renderDuel = () => {
        const revealed = guess !== null;
        const correct = revealed && guess === duel?.realIndex;
        return (
            <>
                <div className="text-center mb-4">
                    <p className="text-muted">
                        Uno es un paciente real (del set de entrenamiento) y el otro lo inventó CTGAN.
                        ¿Puedes distinguirlos? Es el test del discriminador, hecho juego.
                    </p>
                    <div className="d-flex gap-2 justify-content-center flex-wrap">
                        {DISEASE_TABS.map(({ key, label, icon, variant }) => (
                            <Button key={key}
                                variant={gameDisease === key ? variant : `outline-${variant}`}
                                onClick={() => newRound(key)} disabled={loadingDuel}>
                                {icon}{label}
                            </Button>
                        ))}
                    </div>
                </div>

                {loadingDuel && <div className="text-center py-4"><Spinner animation="border" variant="primary" /></div>}

                {duel && !loadingDuel && (
                    <>
                        <Row className="g-3 justify-content-center">
                            {duel.cards.map((card, i) => (
                                <Col md={6} lg={5} key={i}>{renderDuelCard(card, i)}</Col>
                            ))}
                        </Row>

                        {revealed && (
                            <div className="text-center mt-4">
                                <h5 className={correct ? 'text-success' : 'text-danger'}>
                                    {correct ? '¡Acertaste!' : 'Te engañó'}
                                </h5>
                                <p className="text-soft small mx-auto" style={{ maxWidth: '60ch' }}>
                                    {correct
                                        ? 'Distinguiste el caso real del sintético. A veces el sintético se delata por una combinación poco plausible — el límite del método.'
                                        : 'El paciente sintético imitó tan bien la distribución real que pasó por auténtico: justo la propiedad que se busca en un buen generador.'}
                                    {' '}Si no se distinguen, el generador es bueno.
                                </p>
                                <Button variant="dark" onClick={() => newRound()} disabled={loadingDuel}>
                                    <ArrowRepeat className="me-2" />Otra ronda
                                </Button>
                            </div>
                        )}
                    </>
                )}

                {!duel && !loadingDuel && (
                    <div className="text-center text-faint py-4">Elige una enfermedad para empezar una ronda.</div>
                )}
            </>
        );
    };

    const changeDistDisease = (dis) => {
        setDistDisease(dis);
        setDistFeature(DIST_FEATURES[dis][0]);
    };

    const renderDistribution = () => (
        <>
            <div className="text-center mb-4">
                <p className="text-muted">
                    ¿Se parecen de verdad? Aquí se superpone la distribución real y la sintética de una
                    variable: cuanto más se montan las dos áreas, mejor imitó CTGAN a los datos reales.
                </p>
                <div className="d-flex gap-2 justify-content-center flex-wrap mb-3">
                    {DISEASE_TABS.map(({ key, label, icon, variant }) => (
                        <Button key={key}
                            variant={distDisease === key ? variant : `outline-${variant}`}
                            onClick={() => changeDistDisease(key)}>
                            {icon}{label}
                        </Button>
                    ))}
                </div>
                <div className="ps-chips justify-content-center">
                    {DIST_FEATURES[distDisease].map((f) => (
                        <button key={f} type="button"
                            className={`ps-chip ${distFeature === f ? 'on' : ''}`}
                            onClick={() => setDistFeature(f)}>
                            {getLabel(f) || f}
                        </button>
                    ))}
                </div>
            </div>

            {loadingDist && <div className="text-center py-4"><Spinner animation="border" variant="primary" /></div>}

            {distData && !loadingDist && (
                <Card className="border-0 shadow-sm">
                    <Card.Body>
                        <div className="d-flex justify-content-between align-items-center flex-wrap mb-2">
                            <h5 className="mb-0">{getLabel(distFeature) || distFeature}</h5>
                            <span className="text-faint small">Real n={distData.real_n.toLocaleString()} · Sintético n={distData.synthetic_n.toLocaleString()}</span>
                        </div>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={distData.bins} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(127,138,144,.2)" />
                                <XAxis dataKey="bin" tick={{ fontSize: 11, fill: '#7f9296' }} />
                                <YAxis tick={{ fontSize: 11, fill: '#7f9296' }} unit="%" />
                                <Tooltip formatter={(v) => `${v}%`} />
                                <Legend />
                                <Area type="monotone" dataKey="real" name="Real" stroke="#0f6e6a" fill="#16a39c" fillOpacity={0.35} strokeWidth={2} />
                                <Area type="monotone" dataKey="synthetic" name="Sintético" stroke="#c8913a" fill="#d8a64a" fillOpacity={0.3} strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                        <p className="text-soft small mb-0 mt-2">
                            Las áreas son proporciones (cada una suma 100%): la comparación es de forma, no
                            de tamaño de muestra. Eje X: valor de la variable; eje Y: % de pacientes.
                        </p>
                    </Card.Body>
                </Card>
            )}
        </>
    );

    // --- Calidad: heatmap de correlaciones ---
    const shortFeat = (f) => {
        const l = getLabel(f) || f;
        return l.length > 11 ? `${l.slice(0, 10)}…` : l;
    };
    const corrColor = (v) => {
        const a = Math.min(Math.abs(v), 1);
        return v >= 0 ? `rgba(22,163,156,${a})` : `rgba(216,166,74,${a})`;
    };
    const renderHeatmap = (matrix, features, title) => (
        <div className="text-center">
            <div className="fw-bold small mb-2">{title}</div>
            <div style={{ overflowX: 'auto' }}>
                <table style={{ borderCollapse: 'separate', borderSpacing: 2, margin: '0 auto', fontSize: '.7rem' }}>
                    <tbody>
                        <tr>
                            <td />
                            {features.map((f) => (
                                <td key={f} className="text-faint" style={{ padding: '2px', textAlign: 'center' }} title={getLabel(f) || f}>{shortFeat(f)}</td>
                            ))}
                        </tr>
                        {matrix.map((row, i) => (
                            <tr key={i}>
                                <td className="text-faint pe-2 text-end" style={{ whiteSpace: 'nowrap' }} title={getLabel(features[i]) || features[i]}>{shortFeat(features[i])}</td>
                                {row.map((v, j) => (
                                    <td key={j} title={`${v}`}
                                        style={{ background: corrColor(v), color: Math.abs(v) > 0.55 ? '#fff' : 'var(--text)', width: 38, height: 28, textAlign: 'center', borderRadius: 4 }}>
                                        {v.toFixed(1)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );

    const renderQuality = () => (
        <>
            <div className="text-center mb-4">
                <p className="text-muted">
                    ¿Qué tan fiel es el sintético? <strong>SDMetrics</strong> lo compara con el real
                    columna a columna (formas) y en sus relaciones (correlaciones). 1.0 = imitación perfecta.
                </p>
                <div className="d-flex gap-2 justify-content-center flex-wrap">
                    {DISEASE_TABS.map(({ key, label, icon, variant }) => (
                        <Button key={key}
                            variant={qualDisease === key ? variant : `outline-${variant}`}
                            onClick={() => setQualDisease(key)}>
                            {icon}{label}
                        </Button>
                    ))}
                </div>
            </div>

            {loadingQual && <div className="text-center py-4"><Spinner animation="border" variant="primary" /></div>}

            {qualData && !loadingQual && (
                <>
                    <Row className="g-3 mb-4">
                        <Col sm={4}>
                            <div className="ps-kpi text-center">
                                <div className="k-lbl">Fidelidad global</div>
                                <div className="k-val">{qualData.overall != null ? `${(qualData.overall * 100).toFixed(0)}%` : '—'}</div>
                                <div className="k-sub">SDMetrics quality</div>
                            </div>
                        </Col>
                        <Col sm={4}>
                            <div className="ps-kpi text-center">
                                <div className="k-lbl">Distribuciones</div>
                                <div className="k-val">{qualData.column_shapes != null ? `${(qualData.column_shapes * 100).toFixed(0)}%` : '—'}</div>
                                <div className="k-sub">column shapes</div>
                            </div>
                        </Col>
                        <Col sm={4}>
                            <div className="ps-kpi text-center">
                                <div className="k-lbl">Correlaciones</div>
                                <div className="k-val">{qualData.column_pair_trends != null ? `${(qualData.column_pair_trends * 100).toFixed(0)}%` : '—'}</div>
                                <div className="k-sub">pair trends</div>
                            </div>
                        </Col>
                    </Row>

                    {qualData.per_column?.length > 0 && (
                        <Card className="border-0 shadow-sm mb-4">
                            <Card.Body>
                                <h5 className="mb-3">Fidelidad por variable (forma de la distribución)</h5>
                                <Row className="g-2">
                                    {qualData.per_column.map((p) => (
                                        <Col md={6} key={p.column}>
                                            <div className="d-flex align-items-center gap-2">
                                                <span className="small text-soft text-truncate" style={{ width: '128px', flex: 'none' }} title={getLabel(p.column) || p.column}>{getLabel(p.column) || p.column}</span>
                                                <div className="ps-shap-track" style={{ height: '12px' }}>
                                                    <div style={{ width: `${p.score * 100}%`, height: '100%', borderRadius: '6px', background: 'var(--accent)' }} />
                                                </div>
                                                <span className="mono small text-faint" style={{ width: '36px', flex: 'none', textAlign: 'right' }}>{p.score.toFixed(2)}</span>
                                            </div>
                                        </Col>
                                    ))}
                                </Row>
                            </Card.Body>
                        </Card>
                    )}

                    {qualData.corr && (
                        <Card className="border-0 shadow-sm">
                            <Card.Body>
                                <h5 className="mb-1">Correlaciones: ¿preserva las relaciones?</h5>
                                <p className="text-soft small">
                                    Si los dos mapas tienen el mismo patrón de color, CTGAN no solo copió
                                    promedios: mantuvo cómo se relacionan las variables entre sí.
                                </p>
                                <Row className="g-4">
                                    <Col md={6}>{renderHeatmap(qualData.corr.real, qualData.corr.features, 'Real')}</Col>
                                    <Col md={6}>{renderHeatmap(qualData.corr.synthetic, qualData.corr.features, 'Sintético')}</Col>
                                </Row>
                            </Card.Body>
                        </Card>
                    )}

                    <p className="text-faint small text-center mt-3">
                        Calculado con SDMetrics sobre {qualData.n_used?.toLocaleString()} filas submuestreadas de cada conjunto.
                    </p>
                </>
            )}
        </>
    );

    return (
        <Container className="py-5">
            {/* ==============================================
                ENCABEZADO — Caso de estudio (A2 /proyecto)
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mb-5" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Caso de estudio</span>
                <h2>El proyecto por dentro</h2>
                <p>
                    Las decisiones de ingeniería detrás de la herramienta: de dónde salen los
                    datos, cómo se eligen los modelos, cómo se interpreta el resultado y cómo se
                    entrena la IA sin exponer datos de pacientes reales.
                </p>
            </div>

            {/* ==============================================
                DATOS (B1) — de dónde salen, con transparencia
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Los datos</span>
                <h2>De dónde salen los datos</h2>
                <p>
                    Cada enfermedad se entrena con su propia fuente real, curada por separado
                    (sin imputación cruzada): se sanean valores fisiológicamente imposibles y se
                    eligen variables que una persona común puede responder.
                </p>
            </div>

            <Row className="g-3 mb-5">
                <Col md={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-primary">
                                <Droplet className="me-2" />Diabetes
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>100 000</strong> registros de una sola fuente pública, sin
                                valores faltantes. Señal clínica real: glucosa, HbA1c, IMC, edad.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: usa glucosa/HbA1c como variables — uso predictivo
                                legítimo, no fuga de datos.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-danger">
                                <HeartPulse className="me-2" />Hipertensión
                            </div>
                            <p className="text-soft small mb-2">
                                Encuesta de salud (ENSANUT México, <strong>~4 400</strong> personas)
                                con señal real: IMC, cintura y peso correlacionan con el riesgo.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: muestra pequeña y el objetivo es un <em>score</em> de
                                riesgo, no un diagnóstico medido.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-info">
                                <Heart className="me-2" />Cardiovascular
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>~69 000</strong> registros balanceados al 50/50. Se derivan
                                medidas como el IMC a partir de talla y peso.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: presiones imposibles (negativas o de miles) se
                                descartan antes de entrenar.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* ==============================================
                SECCIÓN: TECNOLOGÍA E IA — Datos sintéticos
               ============================================== */}
            <Row className="mb-5 align-items-center">
                <Col lg={7}>
                    <h3 className="mb-3">Tecnología: Datos Sintéticos y Privacidad</h3>
                    <p className="lead text-muted">
                        ¿Cómo entrenamos a la IA sin comprometer la privacidad de los pacientes reales?
                    </p>
                    <p>
                        En salud, usar datos reales es delicado por las leyes de privacidad.
                        Nuestra solución utiliza <strong>Redes Generativas Antagónicas (GAN)</strong>.
                    </p>

                    <Accordion defaultActiveKey="0" className="mb-4">
                        <Accordion.Item eventKey="0">
                            <Accordion.Header><Eyedropper className="me-2" />¿Qué son los Datos Sintéticos?</Accordion.Header>
                            <Accordion.Body>
                                Son registros médicos generados artificialmente que imitan fielmente las estadísticas
                                (promedios, correlaciones) de los pacientes reales, pero no corresponden a ninguna persona física.
                                Esto permite investigar sin riesgos éticos.
                            </Accordion.Body>
                        </Accordion.Item>
                        <Accordion.Item eventKey="1">
                            <Accordion.Header><Robot className="me-2" />¿Cómo funciona una GAN?</Accordion.Header>
                            <Accordion.Body>
                                Es una arquitectura de "competencia" entre dos IAs:
                                <ul>
                                    <li><strong>El Generador:</strong> Intenta crear pacientes falsos creíbles.</li>
                                    <li><strong>El Discriminador:</strong> Intenta distinguir si el paciente es real o falso.</li>
                                </ul>
                                Cuando el discriminador ya no puede notar la diferencia, el modelo está listo para generar datos de alta calidad (CTGAN).
                            </Accordion.Body>
                        </Accordion.Item>
                    </Accordion>
                </Col>

                <Col lg={5}>
                    <Card className="ps-card-accent border-0 shadow">
                        <Card.Body className="p-4">
                            <h5><Lightbulb className="me-2" />Sabías que...</h5>
                            <p className="mb-0">
                                Los modelos de IA de este proyecto fueron entrenados usando una técnica llamada <strong>CTGAN</strong> (Conditional Tabular GAN).
                                Esto permite generar casos raros o extremos para mejorar la capacidad de predicción del sistema.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* DEMO INTERACTIVA — Laboratorio con pestañas */}
            <div className="bg-light p-5 rounded-3 border">
                <div className="text-center mb-4">
                    <h3><Magic className="me-2" />Laboratorio de IA generativa</h3>
                    <p className="text-muted mb-0">
                        Demos interactivas sobre los datos sintéticos que entrenan el sistema.
                    </p>
                </div>

                <Tabs activeKey={labTab} onSelect={(k) => setLabTab(k || 'generar')} id="lab-tabs" className="mb-4 justify-content-center">
                    <Tab eventKey="generar" title={<span><Stars className="me-2" />Generar y evaluar</span>}>
                        <div className="text-center mb-4">
                            <p className="text-muted">
                                Elige una enfermedad y observa cómo la IA "imagina" un paciente con
                                características clínicas coherentes; luego pásalo por el modelo.
                            </p>

                            <div className="d-flex gap-2 justify-content-center flex-wrap mb-3">
                                {DISEASE_TABS.map(({ key, label, icon, variant }) => (
                                    <Button
                                        key={key}
                                        variant={disease === key ? variant : `outline-${variant}`}
                                        onClick={() => generateDemo(key)}
                                        disabled={loading}
                                    >
                                        {icon}{label}
                                    </Button>
                                ))}
                            </div>

                            <Button variant="dark" size="lg" onClick={() => generateDemo()} disabled={loading}>
                                {loading ? (
                                    <span><span className="spinner-border spinner-border-sm me-2"/>Generando...</span>
                                ) : <><Stars className="me-2" />Generar Paciente Sintético</>}
                            </Button>
                        </div>

                        {syntheticData && (
                            <div className="animate__animated animate__fadeInUp">
                                <Row className="justify-content-center">
                                    <Col md={10} lg={8}>
                                        <Card className="shadow-sm border-0">
                                            <Card.Header className="bg-dark text-white d-flex justify-content-between align-items-center">
                                                <span>Perfil Clínico Generado (IA) · {diseaseLabel(syntheticData._disease)}</span>
                                                <Badge bg="warning" text="dark">100% Sintético</Badge>
                                            </Card.Header>
                                            <Table striped hover responsive className="mb-0 text-center align-middle">
                                                <thead className="table-light">
                                                    <tr>
                                                        <th className="text-start ps-4">Variable</th>
                                                        <th>Valor Generado</th>
                                                        <th>Interpretación Rápida</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {buildRows(syntheticData._disease, syntheticData).map((row, i) => (
                                                        <tr key={i}>
                                                            <td className="text-start ps-4 fw-bold">{row.label}</td>
                                                            <td>{row.valor}</td>
                                                            <td>{row.interp}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </Table>
                                            <Card.Footer className="text-center text-muted small bg-white">
                                                Registro creado matemáticamente (CTGAN) a partir de la distribución de
                                                probabilidad de datos clínicos reales. No corresponde a ninguna persona física.
                                            </Card.Footer>
                                        </Card>

                                        <div className="text-center mt-3">
                                            <Button variant="outline-primary" onClick={evaluateSynthetic} disabled={predicting}>
                                                {predicting
                                                    ? <span><span className="spinner-border spinner-border-sm me-2" />Evaluando...</span>
                                                    : <><CpuFill className="me-2" />Evaluar este paciente con el modelo</>}
                                            </Button>
                                            <div className="text-faint small mt-2">
                                                La IA inventó el paciente; ahora el modelo lo evalúa y SHAP explica por qué.
                                            </div>
                                        </div>

                                        {renderPrediction()}
                                    </Col>
                                </Row>
                            </div>
                        )}
                    </Tab>

                    <Tab eventKey="duelo" title={<span><PatchQuestion className="me-2" />¿Real o sintético?</span>}>
                        {renderDuel()}
                    </Tab>

                    <Tab eventKey="dist" title={<span><BarChartLineFill className="me-2" />Distribuciones</span>}>
                        {renderDistribution()}
                    </Tab>

                    <Tab eventKey="calidad" title={<span><ClipboardCheck className="me-2" />Calidad</span>}>
                        {renderQuality()}
                    </Tab>
                </Tabs>
            </div>

            {/* ==============================================
                SELECCIÓN DE MODELOS — panorámica visual → /metricas
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mt-5 mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Modelos</span>
                <h2>Cómo se elige el modelo de cada enfermedad</h2>
                <p>
                    No se asume un único algoritmo: para cada enfermedad compiten tres y se sirve el
                    de mayor AUC en validación cruzada (5-fold). El ganador no es el mismo en las tres.
                </p>
            </div>

            {loadingModels && (
                <div className="text-center py-4"><Spinner animation="border" variant="primary" /></div>
            )}

            {modelsByDisease && (
                <Row className="g-3">
                    {DISEASE_TABS.map(({ key, label, icon, variant }) => {
                        const m = modelsByDisease[key];
                        const board = m?.leaderboard || [];
                        const maxAuc = Math.max(...board.map((r) => r.cv_auc_mean || 0), 0.5001);
                        return (
                            <Col md={4} key={key}>
                                <Card className="h-100 ps-card-hover">
                                    <Card.Body>
                                        <div className="d-flex align-items-center justify-content-between mb-3">
                                            <span className={`fw-bold d-flex align-items-center text-${variant}`}>
                                                {icon}{label}
                                            </span>
                                            {m?.best_model && (
                                                <span className="ps-tag"><TrophyFill className="me-1" />{prettyModel(m.best_model)}</span>
                                            )}
                                        </div>
                                        {board.map((row) => {
                                            const isWinner = row.model === m.best_model;
                                            // Barra proporcional al AUC de CV, con piso en 0.5 (azar)
                                            // para que las diferencias se aprecien sin clavar cifras.
                                            const w = Math.max(6, Math.round(((row.cv_auc_mean - 0.5) / (maxAuc - 0.5)) * 100));
                                            return (
                                                <div className="d-flex align-items-center gap-2 mb-2" key={row.model}>
                                                    <span className="small text-soft" style={{ width: '92px', flex: 'none' }}>{prettyModel(row.model)}</span>
                                                    <div className="ps-shap-track" style={{ height: '14px' }}>
                                                        <div style={{ width: `${w}%`, height: '100%', borderRadius: '7px', background: isWinner ? 'var(--accent)' : 'var(--text-faint)', opacity: isWinner ? 1 : 0.4, transition: 'width .4s ease' }} />
                                                    </div>
                                                    <span className="mono small" style={{ width: '46px', flex: 'none', textAlign: 'right', color: isWinner ? 'var(--accent)' : 'var(--text-faint)', fontWeight: isWinner ? 600 : 400 }}>
                                                        {(row.cv_auc_mean ?? 0).toFixed(3)}
                                                    </span>
                                                </div>
                                            );
                                        })}
                                    </Card.Body>
                                </Card>
                            </Col>
                        );
                    })}
                </Row>
            )}

            <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 mt-4">
                <p className="text-soft small mb-0" style={{ maxWidth: '52ch' }}>
                    Cifras: AUC medio en validación cruzada (5-fold). La probabilidad del simulador es
                    la salida limpia del modelo; los umbrales clínicos (ADA, ACC/AHA) van en una capa
                    aparte.
                </p>
                <Button as={Link} to="/metricas" variant="primary">
                    <BarChartLineFill className="me-2" />
                    Ver el detalle por clase y la desviación
                    <ArrowRight className="ms-2" />
                </Button>
            </div>

            {/* ==============================================
                CAPA CLÍNICA (A4) — interpretación desacoplada
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mt-5 mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Interpretación</span>
                <h2>El modelo no se mezcla con el criterio clínico</h2>
                <p>
                    La probabilidad que ves es la salida limpia del modelo — la misma que respalda
                    el AUC publicado. Los umbrales diagnósticos van en una capa aparte, etiquetada.
                </p>
            </div>

            <Row className="g-3 mb-5 justify-content-center">
                <Col md={6}>
                    <Card className="h-100">
                        <Card.Body>
                            <h5 className="d-flex align-items-center"><CpuFill className="me-2 text-primary" />La probabilidad del modelo</h5>
                            <p className="text-soft small mb-0">
                                Sin retoques ni "pisos" artificiales: lo que el modelo calcula es lo
                                que se muestra y lo que SHAP explica. Así el número es auditable y
                                coherente con las métricas de validación.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={6}>
                    <Card className="h-100">
                        <Card.Body>
                            <h5 className="d-flex align-items-center"><ClipboardPulse className="me-2 text-danger" />La capa de referencia clínica</h5>
                            <p className="text-soft small mb-2">
                                Junto al número, una nota basada en guías reconocidas marca si un valor
                                cruza un umbral diagnóstico — sin alterar la salida del modelo.
                            </p>
                            <div className="ps-clinic">
                                <p className="mb-0">
                                    Glucosa y HbA1c según <strong>ADA</strong>; presión arterial
                                    según <strong>ACC/AHA</strong>.
                                </p>
                                <div className="ref">ADA · ACC/AHA</div>
                            </div>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* ==============================================
                STACK TÉCNICO
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mt-5 mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Stack</span>
                <h2>Con qué está construido</h2>
                <p>
                    Arquitectura desacoplada: una API en Python sirve los modelos; una SPA en
                    React los consume y los visualiza.
                </p>
            </div>

            <Row className="g-3 mb-5">
                <Col md={4}>
                    <Card className="h-100">
                        <Card.Body>
                            <div className="ps-ic"><CodeSlash size={22} /></div>
                            <h5>Backend</h5>
                            <p className="text-soft small mb-3">
                                API REST que carga los modelos, arma el explainer SHAP y expone la
                                capa clínica.
                            </p>
                            <div className="ps-chips">
                                {['Python 3.12', 'Flask', 'scikit-learn', 'LightGBM', 'SHAP'].map((t) => (
                                    <span key={t} className="ps-chip" style={{ cursor: 'default' }}>{t}</span>
                                ))}
                            </div>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100">
                        <Card.Body>
                            <div className="ps-ic"><Magic size={22} /></div>
                            <h5>Datos sintéticos</h5>
                            <p className="text-soft small mb-3">
                                Generación de pacientes virtuales plausibles sin exponer datos reales.
                            </p>
                            <div className="ps-chips">
                                {['SDV', 'CTGAN', 'SDMetrics', 'pandas'].map((t) => (
                                    <span key={t} className="ps-chip" style={{ cursor: 'default' }}>{t}</span>
                                ))}
                            </div>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100">
                        <Card.Body>
                            <div className="ps-ic"><Window size={22} /></div>
                            <h5>Frontend</h5>
                            <p className="text-soft small mb-3">
                                SPA que gestiona la experiencia y renderiza gauges, barras SHAP y
                                gráficos comparativos.
                            </p>
                            <div className="ps-chips">
                                {['React 19', 'Vite', 'React-Bootstrap', 'Recharts'].map((t) => (
                                    <span key={t} className="ps-chip" style={{ cursor: 'default' }}>{t}</span>
                                ))}
                            </div>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* ==============================================
                AUTOR + REPO
               ============================================== */}
            <Card className="mt-5">
                <Card.Body className="p-4 d-flex flex-wrap align-items-center justify-content-between gap-3">
                    <div className="d-flex align-items-center gap-3">
                        <div className="ps-ic" style={{ marginBottom: 0 }}><PersonBadge size={22} /></div>
                        <div>
                            <div className="text-faint" style={{ fontSize: '.76rem', textTransform: 'uppercase', letterSpacing: '.06em', fontWeight: 600 }}>Autor</div>
                            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', lineHeight: 1.15 }}>John Orellana</div>
                            <div className="text-soft small">Predisana · IA explicable en salud</div>
                        </div>
                    </div>
                    <Button href="https://github.com/John-OF/predisana" target="_blank" rel="noopener noreferrer" variant="primary">
                        <Github className="me-2" />Ver el código en GitHub
                    </Button>
                </Card.Body>
            </Card>

        </Container>
    );
};

export default Proyecto;
