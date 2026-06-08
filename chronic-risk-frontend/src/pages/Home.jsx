import { Container, Row, Col, Button, Accordion } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import {
  ArrowRight, InfoCircle, Activity, BarChartLine, ClipboardData,
  Tools, Diagram3, Magic,
} from 'react-bootstrap-icons';

const Home = () => {
  return (
    <div className="cr-page">

      {/* ============ HERO ============ */}
      <header className="ps-hero">
        <Container>
          <Row className="align-items-center g-5">
            <Col lg={7}>
              <span className="ps-eyebrow">● IA explicable para tu salud</span>
              <h1>
                Entiende tu <em>riesgo</em>, no solo tu resultado.
              </h1>
              <p className="lead">
                Predisana estima tu probabilidad de <strong>diabetes</strong>,{' '}
                <strong>hipertensión</strong> y <strong>riesgo cardiovascular</strong> con
                Machine Learning — y te muestra <strong>por qué</strong>, variable por
                variable, con total transparencia.
              </p>
              <div className="d-flex flex-wrap gap-3">
                <Button as={Link} to="/simulacion" variant="primary" size="lg">
                  Probar el simulador <ArrowRight className="ms-1" />
                </Button>
                <Button as={Link} to="/educacion" variant="outline-primary" size="lg">
                  Explorar contenidos
                </Button>
              </div>
              <div className="ps-disclaimer">
                <InfoCircle size={16} />
                Herramienta educativa y demostrativa. No sustituye un diagnóstico médico profesional.
              </div>
            </Col>

            <Col lg={5} className="d-none d-lg-flex justify-content-center">
              <svg className="ps-hero-visual" viewBox="0 0 380 380" fill="none">
                <circle cx="190" cy="190" r="150" stroke="currentColor" strokeWidth="1" opacity=".15" />
                <circle cx="190" cy="190" r="110" stroke="currentColor" strokeWidth="1" opacity=".2" />
                <circle cx="190" cy="190" r="70" stroke="currentColor" strokeWidth="1" opacity=".25" />
                <path
                  d="M40 200 Q90 200 110 200 T150 130 T185 250 T215 175 T245 200 T340 200"
                  stroke="currentColor" strokeWidth="3" fill="none" strokeLinecap="round" opacity=".85"
                />
                <circle cx="245" cy="200" r="6" fill="currentColor" />
              </svg>
            </Col>
          </Row>
        </Container>
      </header>

      {/* ============ 3 CAPACIDADES ============ */}
      <section className="py-4">
        <Container>
          <Row className="g-4">
            <Col md={4}>
              <div className="ps-card ps-card-hover h-100">
                <div className="ps-ic"><Activity size={22} /></div>
                <h3 style={{ fontSize: '1.28rem' }}>Predicción con ML</h3>
                <p className="text-soft mb-0">
                  Modelos entrenados (Regresión Logística, Random Forest, LightGBM) estiman
                  tu riesgo a partir de variables clínicas y de hábitos; se sirve el mejor por
                  enfermedad.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card ps-card-hover h-100">
                <div className="ps-ic"><BarChartLine size={22} /></div>
                <h3 style={{ fontSize: '1.28rem' }}>Explicabilidad SHAP</h3>
                <p className="text-soft mb-0">
                  No es una caja negra: cada predicción viene con un desglose visual de qué
                  factores empujaron tu riesgo hacia arriba o hacia abajo.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card ps-card-hover h-100">
                <div className="ps-ic"><ClipboardData size={22} /></div>
                <h3 style={{ fontSize: '1.28rem' }}>Capa clínica de referencia</h3>
                <p className="text-soft mb-0">
                  Interpretación apoyada en guías ADA y ACC/AHA, presentada aparte y con su
                  contexto — para leer los números con criterio.
                </p>
              </div>
            </Col>
          </Row>
        </Container>
      </section>

      {/* ============ CÓMO FUNCIONA ============ */}
      <section className="py-5">
        <Container>
          <div className="ps-sec-head mx-auto text-center" style={{ maxWidth: '60ch' }}>
            <span className="ps-eyebrow">Metodología</span>
            <h2>¿Cómo funciona Predisana?</h2>
            <p>Del dato crudo a una predicción que puedes interpretar, en tres etapas.</p>
          </div>

          <Row className="g-4">
            <Col md={4}>
              <div className="ps-card h-100">
                <span className="ps-tag">1 · Datos</span>
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Un dataset limpio por enfermedad</h3>
                <p className="text-soft mb-0">
                  Cada enfermedad se entrena con su propia fuente real curada (sin imputación
                  cruzada): saneo de outliers fisiológicos y selección de variables que una
                  persona común <em>puede responder</em>.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card h-100">
                <span className="ps-tag">2 · Modelos</span>
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Selección por validación cruzada</h3>
                <p className="text-soft mb-0">
                  Compite un baseline interpretable (Regresión Logística) contra modelos de
                  boosting y bosques. Se elige el mejor por AUC en cross-validation y se publica
                  el leaderboard completo.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card h-100">
                <span className="ps-tag">3 · Explicación</span>
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Transparente y honesta</h3>
                <p className="text-soft mb-0">
                  La probabilidad mostrada es la salida limpia del modelo, explicada con SHAP.
                  La interpretación clínica (ADA / ACC-AHA) se muestra <em>aparte</em>, sin
                  alterar el número del modelo.
                </p>
              </div>
            </Col>
          </Row>

          <Row className="justify-content-center mt-5">
            <Col lg={10}>
              <h3 className="mb-3" style={{ fontSize: '1.4rem' }}>Profundización técnica</h3>
              <Accordion defaultActiveKey="0">
                <Accordion.Item eventKey="0">
                  <Accordion.Header>
                    <Tools className="me-2" />Arquitectura (Flask + React)
                  </Accordion.Header>
                  <Accordion.Body className="text-soft">
                    Arquitectura desacoplada. El <strong>backend</strong> en Python (Flask) carga
                    los modelos <code>.pkl</code>, procesa las peticiones JSON, construye el
                    explainer SHAP y expone la capa clínica. El <strong>frontend</strong> en
                    React 19 + Vite gestiona la experiencia y renderiza los gráficos.
                  </Accordion.Body>
                </Accordion.Item>
                <Accordion.Item eventKey="1">
                  <Accordion.Header>
                    <Diagram3 className="me-2" />Multi-modelo con selección por CV
                  </Accordion.Header>
                  <Accordion.Body className="text-soft">
                    En lugar de fijar un único algoritmo, se entrena un stack (Regresión
                    Logística, Random Forest, LightGBM) y se sirve el ganador por enfermedad
                    según su AUC en validación cruzada. SHAP se adapta al tipo de modelo
                    (lineal o de árboles).
                  </Accordion.Body>
                </Accordion.Item>
                <Accordion.Item eventKey="2">
                  <Accordion.Header>
                    <Magic className="me-2" />Datos sintéticos (CTGAN)
                  </Accordion.Header>
                  <Accordion.Body className="text-soft">
                    Para alimentar el simulador con casos coherentes sin exponer datos reales,
                    se usa <strong>CTGAN</strong>, que aprende las correlaciones del dataset y
                    genera "pacientes virtuales" estadísticamente plausibles.
                  </Accordion.Body>
                </Accordion.Item>
              </Accordion>
            </Col>
          </Row>
        </Container>
      </section>
    </div>
  );
};

export default Home;
