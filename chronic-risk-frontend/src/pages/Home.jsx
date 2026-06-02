import { Container, Row, Col, Card, Button, Accordion, Badge, ListGroup } from 'react-bootstrap';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="cr-page min-vh-100">

      {/* HERO SECTION */}
      <div className="cr-hero mb-5">
        <Container>
          <Row className="align-items-center">
            <Col lg={7}>
              <div className="hero-badge mb-3">
                <span>Proyecto de Tesis</span>
                <span style={{ opacity: 0.8 }}>•</span>
                <span>Ingeniería de Software</span>
              </div>

              <h1 className="display-4 fw-bold mb-3">
                Inteligencia Artificial Aplicada a la Prevención de Enfermedades Crónicas
              </h1>

              <p className="lead hero-lead mb-4">
                Una plataforma educativa que integra <strong>Machine Learning</strong>,
                <strong> Datos Sintéticos (GAN)</strong> y simulaciones clínicas interactivas para
                entender los factores de riesgo en la salud ecuatoriana.
              </p>

              <div className="d-flex flex-wrap gap-3">
                <Button
                  as={Link}
                  to="/educacion"
                  variant="primary"
                  size="lg"
                  className="px-4 fw-bold shadow-sm"
                >
                  📚 Explorar Biblioteca Educativa
                </Button>

                <Button
                  as={Link}
                  to="/simulacion"
                  variant="success"
                  size="lg"
                  className="px-4 fw-bold shadow-sm"
                >
                  🤖 Probar Simulador de IA
                </Button>
              </div>
            </Col>

            <Col lg={5} className="d-none d-lg-block text-center">
              <div className="hero-illustration">🧬🧠💻</div>
            </Col>
          </Row>
        </Container>
      </div>

      {/* 3 CARDS (como prototipo) */}
      <Container className="pb-5">
        <Row className="g-4 mb-5">
          <Col md={4}>
            <Card className="h-100 cr-action-card hover-effect">
              <Card.Body className="text-center">
                <div className="cr-action-icon mx-auto">📘</div>
                <h5 className="fw-bold">Módulo Educativo</h5>
                <p className="text-muted mb-4">
                  Aprende sobre ENT y factores de riesgo
                </p>
                <Button as={Link} to="/educacion" variant="primary" className="px-4 fw-bold">
                  Ver contenidos
                </Button>
              </Card.Body>
            </Card>
          </Col>

          <Col md={4}>
            <Card className="h-100 cr-action-card hover-effect">
              <Card.Body className="text-center">
                <div className="cr-action-icon mx-auto">🧠</div>
                <h5 className="fw-bold">Simulador con Machine Learning</h5>
                <p className="text-muted mb-4">
                  Genera casos clínicos sintéticos y predice riesgos
                </p>
                <Button as={Link} to="/simulacion" variant="success" className="px-4 fw-bold">
                  Iniciar simulación
                </Button>
              </Card.Body>
            </Card>
          </Col>

          <Col md={4}>
            <Card className="h-100 cr-action-card hover-effect">
              <Card.Body className="text-center">
                <div className="cr-action-icon mx-auto">📊</div>
                <h5 className="fw-bold">Evaluación y Retroalimentación</h5>
                <p className="text-muted mb-4">
                  Mide tu comprensión y evalúa tu experiencia
                </p>
                <Button as={Link} to="/evaluacion" variant="warning" className="px-4 fw-bold">
                  Ir a evaluación
                </Button>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* TODO lo que ya tenías: NO se elimina, sigue igual */}
        <h2 className="text-center mb-5 fw-bold text-secondary">¿Cómo funciona este sistema?</h2>

        <Row className="g-4">
          <Col md={4}>
            <Card className="h-100 border-0 shadow-sm hover-effect">
              <Card.Header className="bg-info text-white fw-bold">1. Ingeniería de Datos</Card.Header>
              <Card.Body>
                <Card.Title>Curación y Normalización</Card.Title>
                <Card.Text className="small text-muted">
                  Antes de la IA, los datos crudos pasan por un proceso riguroso:
                </Card.Text>
                <ListGroup variant="flush" className="small">
                  <ListGroup.Item>📥 Ingesta de múltiples datasets (PIMA, Kaggle, UCI).</ListGroup.Item>
                  <ListGroup.Item>🧹 Limpieza de ceros fisiológicos inválidos (ej. Glucosa = 0).</ListGroup.Item>
                  <ListGroup.Item>🧩 Imputación de valores faltantes usando la <strong>Mediana Estadística</strong>.</ListGroup.Item>
                  <ListGroup.Item>📏 Estandarización de unidades (mg/dL, kg/m²).</ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>

          <Col md={4}>
            <Card className="h-100 border-0 shadow-sm hover-effect">
              <Card.Header className="bg-warning text-dark fw-bold">2. Privacidad con GANs</Card.Header>
              <Card.Body>
                <Card.Title>Generación Sintética</Card.Title>
                <Card.Text className="small text-muted">
                  Para proteger la privacidad de los pacientes reales, utilizamos Inteligencia Artificial Generativa:
                </Card.Text>
                <ListGroup variant="flush" className="small">
                  <ListGroup.Item>🤖 Uso de modelos <strong>CTGAN (Conditional Tabular GAN)</strong>.</ListGroup.Item>
                  <ListGroup.Item>🧠 La IA aprende la distribución estadística real.</ListGroup.Item>
                  <ListGroup.Item>✨ Generación de miles de "pacientes virtuales" matemáticamente coherentes.</ListGroup.Item>
                  <ListGroup.Item>🛡️ Anonimización total: Ningún dato real se expone en la web.</ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>

          <Col md={4}>
            <Card className="h-100 border-0 shadow-sm hover-effect">
              <Card.Header className="bg-success text-white fw-bold">3. Entrenamiento ML</Card.Header>
              <Card.Body>
                <Card.Title>Regresión Logística</Card.Title>
                <Card.Text className="small text-muted">
                  El corazón predictivo del sistema:
                </Card.Text>
                <ListGroup variant="flush" className="small">
                  <ListGroup.Item>⚖️ Algoritmo seleccionado por su transparencia y alta interpretabilidad en salud.</ListGroup.Item>
                  <ListGroup.Item>📈 Pipeline de <strong>Scikit-Learn</strong> con escalado de variables.</ListGroup.Item>
                  <ListGroup.Item>🩺 <strong>Sistema Híbrido:</strong> Combina la predicción matemática con Reglas Clínicas de seguridad.</ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        <hr className="my-5" />

        <Row className="justify-content-center">
          <Col lg={10}>
            <h3 className="mb-4">🔍 Profundización Técnica</h3>
            <Accordion defaultActiveKey="0" className="shadow-sm">
              <Accordion.Item eventKey="0">
                <Accordion.Header>🛠️ Arquitectura del Software (Backend Flask + Frontend React)</Accordion.Header>
                <Accordion.Body>
                  El sistema opera con una arquitectura desacoplada. El <strong>Backend</strong> en Python (Flask) gestiona la lógica pesada:
                  carga los modelos <code>.pkl</code> en memoria, procesa las peticiones JSON y aplica correcciones del sistema experto.
                  El <strong>Frontend</strong> en React.js se encarga de la experiencia de usuario, gestionando el estado de la simulación
                  y renderizando gráficos dinámicos con <code>Recharts</code>.
                </Accordion.Body>
              </Accordion.Item>

              <Accordion.Item eventKey="1">
                <Accordion.Header>⚖️ ¿Por qué Regresión Logística y no Redes Neuronales Profundas?</Accordion.Header>
                <Accordion.Body>
                  En medicina, la <strong>explicabilidad</strong> es vital. Una Regresión Logística permite saber exactamente cuánto peso
                  tiene cada variable. Nuestro enfoque prioriza la transparencia educativa.
                </Accordion.Body>
              </Accordion.Item>

              <Accordion.Item eventKey="2">
                <Accordion.Header>🧬 El reto de los Datos Sintéticos (CTGAN)</Accordion.Header>
                <Accordion.Body>
                  Generar datos de salud es complejo porque las variables están correlacionadas. Usamos <strong>CTGAN</strong> para aprender
                  correlaciones y generar casos coherentes para el simulador.
                </Accordion.Body>
              </Accordion.Item>
            </Accordion>
          </Col>
        </Row>
        {/*
        <div className="text-center mt-5">
          <p className="text-muted">¿Listo para ver la teoría en acción?</p>
          <Button as={Link} to="/simulacion" variant="primary" size="lg" className="rounded-pill px-5 shadow">
            Ir al Simulador
          </Button>
        </div>
        */}
      </Container>
    </div>
  );
};

export default Home;
