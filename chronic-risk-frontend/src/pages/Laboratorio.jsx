import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Container, Row, Col, Card, Button, Table, Badge, Accordion, Spinner, Tabs, Tab } from 'react-bootstrap';
import { getSyntheticCase, evaluateSyntheticCase, getSampleCase, getDistribution, getSyntheticQuality } from '../services/api';
import { getLabel, getFeatureLabel } from '../utils/translations';
import { riskBand, pctRiesgo } from '../utils/riskBand';
import { DISEASE_TABS, diseaseLabel, prettyModel } from '../utils/catalogo';
import { Eyedropper, Robot, Lightbulb, Stars, BarChartLineFill, CpuFill, ArrowUpShort, ArrowDownShort, ArrowRepeat, PatchQuestion, ClipboardCheck } from 'react-bootstrap-icons';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Laboratorio de IA generativa: los datos sinteticos (CTGAN) en cuatro pestanas.
// Vivia a mitad de /proyecto y se pasaba de largo; aqui tiene ruta propia.
const PESTANAS = ['generar', 'duelo', 'dist', 'calidad'];

// Variables numéricas continuas por enfermedad para el histograma comparado.
const DIST_FEATURES = {
    diabetes: ['age', 'bmi', 'hba1c_level', 'waist_circumference', 'bmi_autodeclarado'],
    hipertension: ['age', 'bmi', 'ap_hi', 'waist_circumference', 'bmi_autodeclarado'],
    cardiovascular: ['age', 'bmi', 'ap_hi', 'total_cholesterol', 'egfr'],
    renal: ['age', 'egfr', 'albumin_creatinine_ratio', 'ap_hi', 'bmi_autodeclarado'],
    higado: ['age', 'bmi', 'waist_circumference', 'alt', 'bmi_autodeclarado'],
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

// Presión: manda la más alta de las dos (ACC/AHA 2017).
const presionBadge = (sys, dia) =>
    sys >= 140 || dia >= 90 ? <Badge bg="danger">Hipertensión</Badge> :
    sys >= 130 || dia >= 80 ? <Badge bg="warning">Elevada</Badge> :
    <Badge bg="success">Normal</Badge>;

const hba1cBadge = (v) =>
    v >= 6.5 ? <Badge bg="danger">Diabetes</Badge> :
    v >= 5.7 ? <Badge bg="warning">Prediabetes</Badge> :
    <Badge bg="success">Normal</Badge>;

const colesterolBadge = (total) =>
    total >= 240 ? <Badge bg="danger">Alto</Badge> :
    total >= 200 ? <Badge bg="warning">Límite</Badge> :
    <Badge bg="success">Deseable</Badge>;

// Enfermedad renal crónica (KDIGO): filtrado < 60 o albúmina/creatinina >= 30.
const renalBadge = (egfr, acr) =>
    egfr < 60 || acr >= 300 ? <Badge bg="danger">Alterada</Badge> :
    acr >= 30 ? <Badge bg="warning">Albuminuria</Badge> :
    <Badge bg="success">Normal</Badge>;

// Transaminasa ALT: por encima de ~35 U/L (hombre ~50) se considera elevada.
const altBadge = (alt) =>
    alt >= 80 ? <Badge bg="danger">Muy elevada</Badge> :
    alt >= 40 ? <Badge bg="warning">Elevada</Badge> :
    <Badge bg="success">Normal</Badge>;

const getGenderLabel = (data) => {
    if (data.gender_Male === undefined && data.gender_Female === undefined) return 'No especificado';
    if (Number(data.gender_Male) === 1) return 'Masculino';
    if (Number(data.gender_Female) === 1) return 'Femenino';
    return 'Otro';
};

const tabaco = (d) =>
    Number(d.smoking_history_current) === 1 ? 'Fumador actual' :
    Number(d.smoking_history_former) === 1 ? 'Exfumador' :
    Number(d.smoking_history_never) === 1 ? 'Nunca ha fumado' : '—';

// Diagnósticos previos que trae la ficha (el de la propia enfermedad no está: es el
// objetivo que se estima).
const DIAGNOSTICOS = [
    ['hypertension', 'Hipertensión'], ['diabetes', 'Diabetes'],
    ['high_cholesterol', 'Colesterol alto'], ['heart_disease', 'Cardiovascular'],
];

// Devuelve las filas {label, valor, interp} de una ficha (real o sintética): mismas
// columnas en las dos, clave para el juego "¿cuál es real?". Solo las que trae.
const buildRows = (disease, d) => {
    const hay = (...ks) => ks.every((k) => d[k] !== undefined && d[k] !== null);
    const dx = DIAGNOSTICOS.filter(([k]) => d[k] !== undefined);
    return [
        hay('age') && { label: 'Edad / Sexo', valor: `${Math.floor(d.age)} años / ${getGenderLabel(d)}`, interp: <span className="text-muted">Demográfico</span> },
        hay('bmi') && { label: 'IMC medido', valor: Number(d.bmi).toFixed(1), interp: imcBadge(d.bmi) },
        hay('weight', 'height') && { label: 'Peso / talla (declarados)', valor: `${Math.round(d.weight)} kg / ${Math.round(d.height)} cm`, interp: <span className="text-muted">IMC declarado {Number(d.bmi_autodeclarado).toFixed(1)}</span> },
        hay('waist_circumference') && { label: 'Cintura', valor: `${Math.round(d.waist_circumference)} cm`, interp: <span className="text-muted">Adiposidad central</span> },
        hay('ap_hi', 'ap_lo') && { label: 'Presión', valor: `${Math.round(d.ap_hi)} / ${Math.round(d.ap_lo)} mmHg`, interp: presionBadge(d.ap_hi, d.ap_lo) },
        hay('total_cholesterol', 'hdl_cholesterol') && { label: 'Colesterol total / HDL', valor: `${Math.round(d.total_cholesterol)} / ${Math.round(d.hdl_cholesterol)} mg/dL`, interp: colesterolBadge(d.total_cholesterol) },
        hay('hba1c_level') && { label: 'HbA1c', valor: `${Number(d.hba1c_level).toFixed(1)} %`, interp: hba1cBadge(d.hba1c_level) },
        hay('egfr', 'albumin_creatinine_ratio') && { label: 'Riñón (eGFR / albúmina)', valor: `${Math.round(d.egfr)} / ${Number(d.albumin_creatinine_ratio).toFixed(1)}`, interp: renalBadge(d.egfr, d.albumin_creatinine_ratio) },
        hay('alt') && { label: 'Transaminasa ALT', valor: `${Math.round(d.alt)} U/L`, interp: altBadge(d.alt) },
        dx.length > 0 && {
            label: 'Diagnósticos previos',
            valor: dx.filter(([k]) => Number(d[k]) === 1).map(([, l]) => l).join(', ') || 'Ninguno',
            interp: <span className="text-muted">Autodeclarados</span>,
        },
        { label: 'Tabaco', valor: tabaco(d), interp: <span className="text-muted">Autodeclarado</span> },
    ].filter(Boolean);
};

const Laboratorio = () => {
    const [syntheticData, setSyntheticData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [disease, setDisease] = useState('diabetes');
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);

    // Juego "¿cuál es real?": un duelo real vs sintético.
    const [gameDisease, setGameDisease] = useState(null);   // sin ronda, ninguna marcada
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

    // Pestaña activa (controlada: difiere las cargas pesadas). Va en la URL
    // (?pestana=duelo) para poder enlazar directamente al juego o a la calidad.
    const [params, setParams] = useSearchParams();
    const labTab = PESTANAS.includes(params.get('pestana')) ? params.get('pestana') : 'generar';
    const cambiarPestana = (k) => setParams(k === 'generar' ? {} : { pestana: k }, { replace: true });

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
    // No se registra en la BD: no es una simulación de un usuario.
    const evaluateSynthetic = async () => {
        if (!syntheticData) return;
        setPredicting(true);
        try {
            const { data } = await evaluateSyntheticCase(syntheticData._disease, stripMeta(syntheticData));
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
        const band = riskBand(prediction);  // la misma banda que pinta el Simulador
        const feats = prediction.top_features || [];
        const maxAbs = Math.max(...feats.map((f) => Math.abs(Number(f.shap))), 1e-6);
        return (
            <Card className="border-0 shadow-sm mt-3">
                <Card.Body>
                    <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 mb-3">
                        <div className="d-flex align-items-center gap-3">
                            <div>
                                <div className="text-faint" style={{ fontSize: '.74rem', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>Riesgo estimado</div>
                                <div style={{ fontFamily: 'var(--font-display)', fontSize: '2.1rem', lineHeight: 1.1, color: band.color }}>{pctRiesgo(pct)}</div>
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
                                        <span className="lbl">{getFeatureLabel(f.feature)}</span>
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
                    Tres preguntas distintas sobre el sintético: si <strong>se parece</strong> al real
                    (fidelidad), si <strong>sirve</strong> para entrenar (utilidad) y si <strong>no copia</strong>{' '}
                    a nadie (privacidad). La primera es la fácil; las otras dos son las que mira un revisor.
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
                                                <span className="small text-soft text-truncate" style={{ width: '128px', flex: 'none' }} title={getFeatureLabel(p.column)}>{getFeatureLabel(p.column)}</span>
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

                    {qualData.tstr?.models?.length > 0 && (
                        <Card className="border-0 shadow-sm mb-4">
                            <Card.Body>
                                <h5 className="mb-1">¿Sirve para entrenar? (TSTR)</h5>
                                <p className="text-soft small mb-3">
                                    Se entrena un modelo <strong>solo con datos sintéticos</strong> y se evalúa
                                    contra <strong>pacientes reales que ningún modelo vio al entrenar</strong>. Al lado, el mismo
                                    modelo entrenado con datos reales sobre ese mismo test. Si el sintético fuera
                                    ruido bonito, la barra de abajo se hundiría.
                                </p>
                                {qualData.tstr.models.map((m) => (
                                    <div key={m.model} className="mb-3">
                                        <div className="d-flex justify-content-between small">
                                            <span className="text-soft">{prettyModel(m.model)}</span>
                                            <span className="mono text-faint">
                                                {m.ratio != null ? `${(m.ratio * 100).toFixed(0)}% del real` : '—'}
                                            </span>
                                        </div>
                                        {[
                                            { lbl: 'Entrenado con reales', auc: m.trtr_auc, color: 'var(--text-faint)' },
                                            { lbl: 'Entrenado con sintéticos', auc: m.tstr_auc, color: 'var(--accent)' },
                                        ].map(({ lbl, auc, color }) => (
                                            <div className="d-flex align-items-center gap-2 mt-1" key={lbl}>
                                                <span className="small text-soft" style={{ width: '170px', flex: 'none' }}>{lbl}</span>
                                                <div className="ps-shap-track" style={{ height: '12px' }}>
                                                    <div style={{ width: `${(auc || 0) * 100}%`, height: '100%', borderRadius: '6px', background: color }} />
                                                </div>
                                                <span className="mono small text-faint" style={{ width: '46px', flex: 'none', textAlign: 'right' }}>
                                                    {auc != null ? auc.toFixed(3) : '—'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                ))}
                                <p className="text-faint small mb-0">
                                    AUC sobre {qualData.tstr.n_test_real} pacientes reales del test. Mismo algoritmo
                                    en las dos ramas, para que la diferencia sea de los datos y no del modelo.
                                </p>
                            </Card.Body>
                        </Card>
                    )}

                    {qualData.privacy && (
                        <Card className="border-0 shadow-sm mb-4">
                            <Card.Body>
                                <h5 className="mb-1">¿Copia a alguien? (privacidad)</h5>
                                <p className="text-soft small mb-3">
                                    Un GAN que memoriza deja de anonimizar. Se mide cuánto dista cada ficha
                                    sintética del paciente real más parecido. La referencia honesta no es cero:
                                    <strong> dos muestras reales distintas también se parecen</strong>, así que se
                                    compara con lo que dista el test real.
                                </p>
                                <Row className="g-3">
                                    <Col sm={4}>
                                        <div className="ps-kpi text-center">
                                            <div className="k-lbl">Distancia del sintético</div>
                                            <div className="k-val">{qualData.privacy.median_synthetic}</div>
                                            <div className="k-sub">al paciente real más cercano</div>
                                        </div>
                                    </Col>
                                    <Col sm={4}>
                                        <div className="ps-kpi text-center">
                                            <div className="k-lbl">Referencia: test real</div>
                                            <div className="k-val">{qualData.privacy.median_real_test}</div>
                                            <div className="k-sub">personas reales distintas</div>
                                        </div>
                                    </Col>
                                    <Col sm={4}>
                                        <div className="ps-kpi text-center">
                                            <div className="k-lbl">Copias exactas</div>
                                            <div className="k-val">{qualData.privacy.exact_copies}</div>
                                            <div className="k-sub">de {qualData.privacy.n_synthetic} fichas</div>
                                        </div>
                                    </Col>
                                </Row>
                                <p className="text-faint small mt-3 mb-0">
                                    {qualData.privacy.ratio >= 1 ? (
                                        <>
                                            El sintético dista de los datos de entrenamiento <strong>{qualData.privacy.ratio} veces</strong> lo
                                            que dista el propio test real: no se parece a los reales <em>más</em> de lo
                                            que se parecen entre sí dos muestras reales.
                                        </>
                                    ) : (
                                        <>
                                            Ojo: el sintético queda <strong>más cerca</strong> de los datos de entrenamiento
                                            que el propio test real ({qualData.privacy.ratio} veces su distancia).
                                        </>
                                    )}
                                    {qualData.privacy.exact_copies_real_test > 0 && (
                                        <>
                                            {' '}Las coincidencias exactas tampoco son señal de copia aquí: entre los
                                            propios datos reales hay {qualData.privacy.exact_copies_real_test} de{' '}
                                            {qualData.privacy.n_real_test}, porque las variables son gruesas y dos
                                            personas distintas comparten ficha a menudo.
                                        </>
                                    )}
                                </p>
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
                        Parte de este parecido es por construcción, no aprendido por la GAN: las proporciones
                        de lo categórico, la obesidad grave, el tope de edad 80 y la regla de laboratorio que
                        define la enfermedad se imponen al elegir las fichas, y las relaciones más fuertes
                        (peso e IMC, cintura, presión) se reconstruyen con una recta del real.
                    </p>
                </>
            )}
        </>
    );

    return (
        <Container className="py-5">
            <div className="ps-sec-head mx-auto text-center mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Laboratorio</span>
                <h2>Laboratorio de IA generativa</h2>
                <p>
                    Pacientes que no existen, generados por una GAN a partir de los datos de NHANES.
                    Genéralos y pásalos por el modelo, intenta distinguirlos de los reales, compara sus
                    distribuciones y mide si se parecen, si sirven y si copian a alguien.
                </p>
            </div>

            {/* DEMO INTERACTIVA — Laboratorio con pestañas */}
            <div className="bg-light p-3 p-md-5 rounded-3 border mb-5">

                <Tabs activeKey={labTab} onSelect={(k) => cambiarPestana(k || 'generar')} id="lab-tabs" className="mb-4 justify-content-center">
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
                SECCIÓN: TECNOLOGÍA E IA — Datos sintéticos
               ============================================== */}
            <Row className="mb-5 align-items-center">
                <Col lg={7}>
                    <h3 className="mb-3">Qué son y por qué importan</h3>
                    <p className="lead text-muted">
                        ¿Se puede trabajar con datos de salud sin exponer a pacientes reales?
                    </p>
                    <p>
                        En salud, compartir datos reales es delicado por las leyes de privacidad. Los
                        modelos del simulador se entrenan con NHANES, una encuesta pública que los CDC
                        publican anonimizada; este laboratorio explora la otra vía: las{' '}
                        <strong>Redes Generativas Antagónicas (GAN)</strong>, que crean pacientes que no existen.
                    </p>

                    <Accordion defaultActiveKey="0" className="mb-4">
                        <Accordion.Item eventKey="0">
                            <Accordion.Header><Eyedropper className="me-2" />¿Qué son los Datos Sintéticos?</Accordion.Header>
                            <Accordion.Body>
                                Son registros médicos generados artificialmente que imitan fielmente las estadísticas
                                (promedios, correlaciones) de los pacientes reales, pero no corresponden a ninguna persona física.
                                Así se puede compartir y experimentar con mucho menos riesgo para la privacidad, siempre que
                                se compruebe que no copian a nadie (pestaña Calidad).
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
                                Los modelos del simulador se entrenan con los datos reales, no con el sintético.
                                El sintético (<strong>CTGAN</strong>, Conditional Tabular GAN) alimenta este laboratorio y
                                una prueba: entrenar un modelo solo con él y evaluarlo con pacientes reales (TSTR, en
                                la pestaña Calidad).
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default Laboratorio;
