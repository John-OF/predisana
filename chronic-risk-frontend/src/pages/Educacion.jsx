import { useState } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, Accordion, Tab, Nav, ListGroup } from 'react-bootstrap';
import { getSyntheticCase } from '../services/api';

const INFO_ENFERMEDADES = {
    diabetes: {
        titulo: "Diabetes Tipo 2",
        color: "primary",
        definicion: "Trastorno metabólico crónico caracterizado por niveles elevados de glucosa en sangre, debido a que el cuerpo no produce suficiente insulina o no la utiliza eficazmente.",
        mortalidad: "En Ecuador, la diabetes mellitus es una de las principales causas de muerte no violenta.",
        poblacion: "Se estima que afecta a cerca del 5.5% de la población (aprox. 1 de cada 18 ecuatorianos).",
        organos: ["Páncreas", "Riñones (Nefropatía)", "Ojos (Retina)", "Corazón", "Nervios periféricos"],
        variables_ia: "El modelo analiza Glucosa, HbA1c y BMI porque son indicadores directos del estado metabólico.",
        factores: [
            "Sedentarismo y falta de actividad física.",
            "Dieta alta en carbohidratos refinados y azúcares.",
            "Antecedentes familiares (Genética).",
            "Obesidad abdominal."
        ],
        sintomas: ["Sed excesiva (Polidipsia)", "Micción frecuente", "Visión borrosa", "Fatiga crónica", "Cicatrización lenta"],
        tratamiento: "No tiene cura definitiva, pero es controlable. Requiere monitoreo de glucosa, dieta balanceada, ejercicio y medicación (Insulina o antidiabéticos)."
    },
    hipertension: {
        titulo: "Hipertensión Arterial",
        color: "danger",
        definicion: "Afección en la que la fuerza que ejerce la sangre contra las paredes de las arterias es lo suficientemente alta como para causar problemas de salud cardíaca.",
        mortalidad: "Conocida como 'el asesino silencioso', es un factor crítico en infartos y accidentes cerebrovasculares.",
        poblacion: "Afecta aproximadamente al 19.8% de la población adulta en Ecuador.",
        organos: ["Corazón (Insuficiencia)", "Arterias", "Cerebro", "Riñones"],
        variables_ia: "La presión arterial sistólica/diastólica y la edad son las variables de mayor peso en la predicción.",
        factores: [
            "Consumo excesivo de sal (Sodio).",
            "Estrés crónico.",
            "Consumo de tabaco y alcohol.",
            "Edad avanzada."
        ],
        sintomas: ["Generalmente asintomática.", "Dolor de cabeza matutino.", "Zumbido en oídos (Tinnitus).", "Sangrado nasal (casos graves)."],
        tratamiento: "Crónico y de por vida. Se trata reduciendo el sodio, controlando el peso y usando fármacos antihipertensivos."
    },
    cardiovascular: {
        titulo: "Riesgo Cardiovascular",
        color: "info",
        definicion: "Grupo de desórdenes del corazón y los vasos sanguíneos, incluyendo cardiopatías coronarias y enfermedades cerebrovasculares.",
        mortalidad: "Es la causa #1 de muerte a nivel mundial y en Ecuador (aprox. 24% de defunciones).",
        poblacion: "El riesgo aumenta significativamente en hombres >45 años y mujeres >55 años.",
        organos: ["Corazón", "Cerebro", "Sistema circulatorio completo"],
        variables_ia: "El modelo cruza datos de tabaquismo, colesterol y presión arterial para estimar este riesgo.",
        factores: [
            "Tabaquismo (Factor crítico).",
            "Colesterol LDL alto.",
            "Diabetes e Hipertensión no controladas.",
            "Inactividad física."
        ],
        sintomas: ["Dolor u opresión en el pecho (Angina)", "Falta de aire", "Entumecimiento en extremidades", "Palpitaciones"],
        tratamiento: "Prevención primaria (estilo de vida) y secundaria (fármacos como estatinas, aspirina, cirugías)."
    }
};

const Educacion = () => {
    const [syntheticData, setSyntheticData] = useState(null);
    const [loading, setLoading] = useState(false);

    const generateDemo = async () => {
        setLoading(true);
        try {
            const { data } = await getSyntheticCase('diabetes');
            setSyntheticData(data);
        } catch (error) {
            console.error("Error generando caso:", error);
        } finally {
            setLoading(false);
        }
    };

    // --- Helpers de visualización ---
    const getGenderLabel = (data) => {
        if (data.gender_Male === undefined && data.gender_Female === undefined) return 'No especificado';
        const isMale = Number(data.gender_Male) === 1;
        const isFemale = Number(data.gender_Female) === 1;
        if (isMale) return 'Masculino';
        if (isFemale) return 'Femenino';
        return 'Otro';
    };

    const getBloodPressure = (data) => {
        const bp = data.blood_pressure || data.ap_hi;
        if (bp && bp > 0 && bp !== 120) return Math.round(bp);
        if (bp === 120) return 120;
        return null;
    };

    return (
        <Container className="py-5">
            {/* ==============================================
                SECCIÓN 1: ENCICLOPEDIA MÉDICA (Requisito Tutor)
               ============================================== */}
            <div className="mb-5">
                <div className="text-center mb-5">
                    <h2 className="fw-bold text-primary">Biblioteca de Enfermedades Crónicas</h2>
                    <p className="lead text-muted">
                        Entendiendo las patologías que afectan a la población general y sus factores de riesgo.
                    </p>
                </div>

                <Tab.Container id="medical-info-tabs" defaultActiveKey="diabetes">
                    <Row>
                        <Col sm={3} className="mb-4">
                            <Nav variant="pills" className="flex-column shadow-sm p-3 bg-white rounded">
                                <Nav.Item>
                                    <Nav.Link eventKey="diabetes">🩺 Diabetes T2</Nav.Link>
                                </Nav.Item>
                                <Nav.Item>
                                    <Nav.Link eventKey="hipertension">💓 Hipertensión</Nav.Link>
                                </Nav.Item>
                                <Nav.Item>
                                    <Nav.Link eventKey="cardiovascular">🚑 Cardiovascular</Nav.Link>
                                </Nav.Item>
                            </Nav>
                        </Col>
                        
                        <Col sm={9}>
                            <Tab.Content>
                                {Object.entries(INFO_ENFERMEDADES).map(([key, info]) => (
                                    <Tab.Pane eventKey={key} key={key}>
                                        <Card className={`border-${info.color} shadow-sm h-100`}>
                                            <Card.Header className={`bg-${info.color} text-white fw-bold`}>
                                                {info.titulo}
                                            </Card.Header>
                                            <Card.Body>
                                                <Card.Title>Definición</Card.Title>
                                                <Card.Text className="mb-4">{info.definicion}</Card.Text>

                                                <Row className="mb-4">
                                                    <Col md={6}>
                                                        <div className="p-3 bg-light rounded h-100">
                                                            <h6 className="text-danger fw-bold">📉 Impacto en Ecuador</h6>
                                                            <ul className="small mb-0 ps-3">
                                                                <li><strong>Mortalidad:</strong> {info.mortalidad}</li>
                                                                <li className="mt-2"><strong>Población Afectada:</strong> {info.poblacion}</li>
                                                            </ul>
                                                        </div>
                                                    </Col>
                                                    <Col md={6}>
                                                        <div className="p-3 bg-light rounded h-100">
                                                            <h6 className="text-primary fw-bold">🧠 Variables Clave (IA)</h6>
                                                            <p className="small mb-0">{info.variables_ia}</p>
                                                        </div>
                                                    </Col>
                                                </Row>

                                                <Row>
                                                    <Col md={6}>
                                                        <h6>🚫 Factores de Riesgo / Hábitos</h6>
                                                        <ListGroup variant="flush" className="small">
                                                            {info.factores.map((f, i) => (
                                                                <ListGroup.Item key={i} className="px-0 py-1">
                                                                    • {f}
                                                                </ListGroup.Item>
                                                            ))}
                                                        </ListGroup>
                                                    </Col>
                                                    <Col md={6}>
                                                        <h6>⚠️ Principales Síntomas</h6>
                                                        <ListGroup variant="flush" className="small">
                                                            {info.sintomas.map((s, i) => (
                                                                <ListGroup.Item key={i} className="px-0 py-1">
                                                                    • {s}
                                                                </ListGroup.Item>
                                                            ))}
                                                        </ListGroup>
                                                    </Col>
                                                </Row>

                                                <hr />
                                                
                                                <div className="d-flex justify-content-between align-items-start flex-wrap">
                                                    <div className="mb-2">
                                                        <strong>Órganos Afectados: </strong>
                                                        {info.organos.map((org, i) => (
                                                            <Badge bg="secondary" className="me-1" key={i}>{org}</Badge>
                                                        ))}
                                                    </div>
                                                    <div style={{maxWidth: '400px'}}>
                                                        <strong>Tratamiento: </strong> 
                                                        <span className="text-muted small">{info.tratamiento}</span>
                                                    </div>
                                                </div>
                                            </Card.Body>
                                        </Card>
                                    </Tab.Pane>
                                ))}
                            </Tab.Content>
                        </Col>
                    </Row>
                </Tab.Container>
            </div>

            <hr className="my-5" />

            {/* ==============================================
                SECCIÓN 2: TECNOLOGÍA E IA (Lo existente mejorado)
               ============================================== */}
            
            <Row className="mb-5 align-items-center">
                <Col lg={7}>
                    <h3 className="mb-3 text-dark">Tecnología: Datos Sintéticos y Privacidad</h3>
                    <p className="lead text-muted">
                        ¿Cómo entrenamos a la IA sin comprometer la privacidad de los pacientes reales?
                    </p>
                    <p>
                        En salud, usar datos reales es delicado por las leyes de privacidad. 
                        Nuestra solución utiliza <strong>Redes Generativas Antagónicas (GAN)</strong>.
                    </p>
                    
                    <Accordion defaultActiveKey="0" className="mb-4">
                        <Accordion.Item eventKey="0">
                            <Accordion.Header>🧪 ¿Qué son los Datos Sintéticos?</Accordion.Header>
                            <Accordion.Body>
                                Son registros médicos generados artificialmente que imitan fielmente las estadísticas 
                                (promedios, correlaciones) de los pacientes reales, pero no corresponden a ninguna persona física. 
                                Esto permite investigar sin riesgos éticos.
                            </Accordion.Body>
                        </Accordion.Item>
                        <Accordion.Item eventKey="1">
                            <Accordion.Header>🤖 ¿Cómo funciona una GAN?</Accordion.Header>
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
                    <Card className="bg-primary text-white border-0 shadow">
                        <Card.Body className="p-4">
                            <h5>💡 Sabías que...</h5>
                            <p className="mb-0">
                                Los modelos de IA de este proyecto fueron entrenados usando una técnica llamada <strong>CTGAN</strong> (Conditional Tabular GAN). 
                                Esto permite generar casos raros o extremos para mejorar la capacidad de predicción del sistema.
                            </p>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* DEMO INTERACTIVA */}
            <div className="bg-light p-5 rounded-3 border">
                <div className="text-center mb-4">
                    <h3>🧬 Laboratorio de Generación (Demo GAN)</h3>
                    <p className="text-muted">
                        Observa cómo la IA es capaz de "imaginar" un paciente con características clínicas coherentes.
                    </p>
                    <Button 
                        variant="dark" 
                        size="lg" 
                        onClick={generateDemo} 
                        disabled={loading}
                    >
                        {loading ? (
                            <span><span className="spinner-border spinner-border-sm me-2"/>Generando...</span>
                        ) : '✨ Generar Paciente Sintético'}
                    </Button>
                </div>

                {syntheticData && (
                    <div className="animate__animated animate__fadeInUp">
                        <Row className="justify-content-center">
                            <Col md={10} lg={8}>
                                <Card className="shadow-sm border-0">
                                    <Card.Header className="bg-dark text-white d-flex justify-content-between align-items-center">
                                        <span>Perfil Clínico Generado (IA)</span>
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
                                            <tr>
                                                <td className="text-start ps-4 fw-bold">Edad / Género</td>
                                                <td>{Math.floor(syntheticData.age)} años / {getGenderLabel(syntheticData)}</td>
                                                <td>Demográfico</td>
                                            </tr>
                                            <tr>
                                                <td className="text-start ps-4 fw-bold">IMC (Masa Corporal)</td>
                                                <td>{Number(syntheticData.bmi).toFixed(1)}</td>
                                                <td>
                                                    {syntheticData.bmi >= 30 ? <Badge bg="danger">Obesidad</Badge> : 
                                                     syntheticData.bmi >= 25 ? <Badge bg="warning">Sobrepeso</Badge> : 
                                                     <Badge bg="success">Normal</Badge>}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td className="text-start ps-4 fw-bold">Glucosa</td>
                                                <td className={syntheticData.glucose > 140 ? 'fw-bold text-danger' : ''}>
                                                    {Math.round(syntheticData.glucose)} mg/dL
                                                </td>
                                                <td>
                                                    {syntheticData.glucose > 200 ? <Badge bg="danger">Diabetes</Badge> : 
                                                     syntheticData.glucose > 100 ? <Badge bg="warning">Riesgo</Badge> : 
                                                     <Badge bg="success">Normal</Badge>}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td className="text-start ps-4 fw-bold">Presión Arterial</td>
                                                <td>
                                                    {getBloodPressure(syntheticData) ? `${getBloodPressure(syntheticData)} mmHg` : 'N/A'}
                                                </td>
                                                <td>
                                                    {getBloodPressure(syntheticData) >= 140 ? <Badge bg="danger">Hipertensión</Badge> :
                                                     getBloodPressure(syntheticData) >= 130 ? <Badge bg="warning">Elevada</Badge> : 
                                                     '-'}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td className="text-start ps-4 fw-bold">Enfermedad Cardíaca</td>
                                                <td>{Number(syntheticData.heart_disease) === 1 ? 'Sí' : 'No'}</td>
                                                <td>
                                                    {Number(syntheticData.heart_disease) === 1 && <Badge bg="danger">Historial Presente</Badge>}
                                                </td>
                                            </tr>
                                        </tbody>
                                    </Table>
                                    <Card.Footer className="text-center text-muted small bg-white">
                                        Este registro fue creado matemáticamente basándose en la distribución de probabilidad de datos reales de Ecuador.
                                    </Card.Footer>
                                </Card>
                            </Col>
                        </Row>
                    </div>
                )}
            </div>
        </Container>
    );
};

export default Educacion;