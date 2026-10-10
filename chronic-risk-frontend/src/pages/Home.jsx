import { Container, Row, Col, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import {
  ArrowRight, InfoCircle, Activity, BarChartLine, ClipboardData,
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
                <strong>hipertensión</strong>, <strong>riesgo cardiovascular</strong>,{' '}
                <strong>enfermedad renal crónica</strong> e <strong>hígado graso</strong> con
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
                  Dos modelos por enfermedad: uno simplificado, con lo que cualquiera sabe de sí
                  mismo, y uno completo, con medidas y análisis. Estiman la enfermedad total,
                  diagnosticada o no.
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
                  Interpretación apoyada en guías ADA, ACC/AHA y KDIGO, presentada aparte y con su
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
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Datos reales, revisados</h3>
                <p className="text-soft mb-0">
                  Las cinco enfermedades salen de NHANES 2017-2023 (CDC), con examen físico y
                  laboratorio. Antes de entrenar se limpian, y el modo simplificado usa el peso y
                  la talla que la gente <em>declara</em>, que es lo que escribirá aquí.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card h-100">
                <span className="ps-tag">2 · Modelos</span>
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Selección con filtro clínico</h3>
                <p className="text-soft mb-0">
                  Compiten Regresión Logística, LightGBM y Random Forest, ajustados por validación
                  cruzada anidada. Solo gana quien respeta el sentido clínico de cada variable, y se
                  publica el leaderboard completo.
                </p>
              </div>
            </Col>
            <Col md={4}>
              <div className="ps-card h-100">
                <span className="ps-tag">3 · Explicación</span>
                <h3 style={{ fontSize: '1.2rem', margin: '12px 0 8px' }}>Transparente y honesta</h3>
                <p className="text-soft mb-0">
                  La probabilidad mostrada es la del modelo, solo calibrada (sin cambiar su orden),
                  y SHAP explica qué la movió. La interpretación clínica (ADA, ACC/AHA, KDIGO) se
                  muestra <em>aparte</em>, sin alterar el número del modelo.
                </p>
              </div>
            </Col>
          </Row>

          <div className="text-center mt-5">
            <p className="text-soft mb-3">
              ¿Te interesa cómo se construyó? El caso de estudio explica los datos, la
              selección de modelos, la capa clínica y el stack técnico.
            </p>
            <Button as={Link} to="/proyecto" variant="outline-primary" size="lg">
              Acerca de Predisana <ArrowRight className="ms-1" />
            </Button>
          </div>
        </Container>
      </section>
    </div>
  );
};

export default Home;
