import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Container, Row, Col, Card, Button, Spinner } from 'react-bootstrap';
import { getMetrics } from '../services/api';
import { DISEASE_TABS, prettyModel } from '../utils/catalogo';
import { Droplet, HeartPulse, Heart, Funnel, Activity, Magic, BarChartLineFill, ArrowRight, TrophyFill, CpuFill, ClipboardPulse, CodeSlash, Window, Github, PersonBadge } from 'react-bootstrap-icons';

const Proyecto = () => {
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
                    datos, cómo se eligen los modelos, cómo se interpreta el resultado y con qué
                    está construido.
                </p>
            </div>

            {/* ==============================================
                DATOS (B1) — de dónde salen, con transparencia
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Los datos</span>
                <h2>De dónde salen los datos</h2>
                <p>
                    Las cinco enfermedades salen de <strong>NHANES 2017-2023</strong>, la encuesta de salud
                    de los CDC con examen físico y laboratorio. Antes de entrenar se revisan y limpian los
                    datos: fuera las embarazadas, un «no sabe» cuenta como dato faltante y no como un «no»,
                    y fuera las variables cuyo cuestionario cambió entre ciclos.
                </p>
            </div>

            <Row className="g-3 mb-5">
                <Col md={6} lg={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-primary">
                                <Droplet className="me-2" />Diabetes
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>14 347</strong> adultos con HbA1c medida. Se estima la diabetes
                                total: diagnosticada o detectada por análisis (HbA1c ≥ 6,5% o glucosa en
                                ayunas ≥ 126). Más de 1 de cada 5 diabéticos no estaba diagnosticado.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: la HbA1c y la glucosa <strong>no</strong> son variables del
                                modelo: definen la enfermedad. Si se aportan, las interpreta la guía (ADA).
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={6} lg={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-danger">
                                <HeartPulse className="me-2" />Hipertensión
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>14 017</strong> adultos con la presión medida. Se estima la
                                hipertensión total: diagnosticada, ≥ 140/90 mmHg medida o con medicación (44%).
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: la presión medida <strong>no</strong> es una variable del
                                modelo — sería un umbral disfrazado. Se interpreta aparte, con la referencia ACC/AHA.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={6} lg={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-info">
                                <Heart className="me-2" />Cardiovascular
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>16 815</strong> adultos. Se estima la enfermedad cardiovascular
                                diagnosticada: coronaria, angina, infarto, insuficiencia cardiaca o ictus (13%).
                                Sustituye al dataset de Kaggle, de origen poco documentado.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: en el modo completo, la presión, el colesterol y la HbA1c no
                                tienen efecto — quien ya tuvo un evento suele estar en tratamiento — y la app lo avisa.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={6} lg={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-success">
                                <Funnel className="me-2" />Renal crónica
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>13 532</strong> adultos con análisis de sangre y de orina. Se estima la
                                enfermedad total: diagnosticada, filtrado glomerular &lt; 60 o albúmina en orina
                                ≥ 30 mg/g (19%). Solo 1 de cada 5 estaba diagnosticado.
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: el filtrado y la albúmina <strong>no</strong> son variables del
                                modelo: definen la enfermedad. Si se aportan, las interpreta la guía (KDIGO), que
                                pide además que se mantengan 3 meses.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={6} lg={4}>
                    <Card className="h-100 ps-card-hover">
                        <Card.Body>
                            <div className="d-flex align-items-center mb-2 fw-bold text-warning">
                                <Activity className="me-2" />Hígado graso
                            </div>
                            <p className="text-soft small mb-2">
                                <strong>13 292</strong> adultos con una elastografía (FibroScan) válida. Se estima
                                el hígado graso medido: CAP ≥ 288 dB/m (34%). No separa la causa (alcohol u otras).
                            </p>
                            <p className="text-faint small mb-0">
                                Transparencia: lo que lo define, el CAP, casi nadie lo tiene a mano, así que no se
                                pide: el modelo lo estima con los factores metabólicos y, en el completo, la
                                transaminasa ALT. Solo una elastografía lo confirma.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* ==============================================
                DATOS SINTÉTICOS — el laboratorio vive en /laboratorio
               ============================================== */}
            <Card className="ps-card-accent border-0 shadow mb-5">
                <Card.Body className="p-4 d-flex flex-wrap align-items-center justify-content-between gap-3">
                    <div style={{ maxWidth: '64ch' }}>
                        <h5 className="mb-2"><Magic className="me-2" />Datos sintéticos</h5>
                        <p className="mb-0">
                            Los modelos se entrenan con los datos reales. Aparte, una GAN (CTGAN) genera
                            pacientes que no existen: en el laboratorio puedes generarlos y evaluarlos,
                            jugar a distinguirlos de los reales y ver cuánto se parecen y si copian a alguien.
                        </p>
                    </div>
                    <Button as={Link} to="/laboratorio" variant="outline-light" className="fw-semibold">
                        Ir al laboratorio <ArrowRight className="ms-1" />
                    </Button>
                </Card.Body>
            </Card>


            {/* ==============================================
                SELECCIÓN DE MODELOS — panorámica visual → /metricas
               ============================================== */}
            <div className="ps-sec-head mx-auto text-center mt-5 mb-4" style={{ maxWidth: '62ch' }}>
                <span className="ps-eyebrow">Modelos</span>
                <h2>Cómo se elige el modelo de cada enfermedad</h2>
                <p>
                    No se asume un único algoritmo: para cada enfermedad compiten tres, cada uno con sus
                    parámetros ajustados por validación cruzada anidada. Solo puede ganar quien pasa el
                    filtro de validación: ninguna variable puede mover el riesgo al revés de su sentido
                    clínico. Random Forest, que no admite esa restricción, no lo pasa en ninguna.
                    (Modo simplificado; el completo, en Métricas.)
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
                            <Col md={6} lg={4} key={key}>
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
                                            const fuera = row.pasa_filtro === false;
                                            // Barra proporcional al AUC de CV, con piso en 0.5 (azar)
                                            // para que las diferencias se aprecien sin clavar cifras.
                                            const w = Math.max(6, Math.round(((row.cv_auc_mean - 0.5) / (maxAuc - 0.5)) * 100));
                                            return (
                                                <div className="d-flex align-items-center gap-2 mb-2" key={row.model}
                                                    title={fuera ? `No pasa el filtro: ${(row.motivos || []).join('; ')}` : undefined}>
                                                    <span className="small text-soft" style={{ width: '92px', flex: 'none', textDecoration: fuera ? 'line-through' : 'none' }}>{prettyModel(row.model)}</span>
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
                    Cifras: AUC medio en validación cruzada anidada (5 pliegues). La probabilidad del simulador es
                    la del modelo, calibrada; los umbrales clínicos (ADA, ACC/AHA, KDIGO) van en una capa
                    aparte.
                </p>
                <Button as={Link} to="/metricas" variant="primary">
                    <BarChartLineFill className="me-2" />
                    Ver el detalle por clase y el modo completo
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
                    La probabilidad que ves es la del modelo, calibrada sin cambiar su orden: el AUC
                    publicado es el suyo. Los umbrales diagnósticos van en una capa aparte, etiquetada.
                </p>
            </div>

            <Row className="g-3 mb-5 justify-content-center">
                <Col md={6}>
                    <Card className="h-100">
                        <Card.Body>
                            <h5 className="d-flex align-items-center"><CpuFill className="me-2 text-primary" />La probabilidad del modelo</h5>
                            <p className="text-soft small mb-0">
                                Sin retoques ni "pisos" artificiales: solo una calibración isotónica,
                                para que un 30% signifique de verdad un 30%, que no cambia el orden de
                                los casos (el AUC es el mismo). SHAP explica la salida del modelo antes
                                de calibrar. Así el número es auditable y coherente con las métricas.
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
                                    según <strong>ACC/AHA</strong>; función renal según <strong>KDIGO</strong>.
                                </p>
                                <div className="ref">ADA · ACC/AHA · KDIGO</div>
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
