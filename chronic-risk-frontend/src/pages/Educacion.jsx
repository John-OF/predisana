import { useState } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, Accordion, Tab, Nav, ListGroup } from 'react-bootstrap';
import { getSyntheticCase } from '../services/api';
import { Droplet, HeartPulse, Heart, GraphDownArrow, Cpu, ExclamationOctagon, ExclamationTriangle, Eyedropper, Robot, Lightbulb, Magic, Stars } from 'react-bootstrap-icons';

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

// Cada enfermedad tiene un esquema de datos distinto (diabetes no mide presión
// continua, cardiovascular usa colesterol/glucosa ordinales 1-3, hipertensión
// trae cintura/peso...). Por eso la interpretación clínica es por-enfermedad.
const DISEASE_TABS = [
    { key: 'diabetes', label: 'Diabetes', icon: <Droplet className="me-2" />, variant: 'primary' },
    { key: 'hipertension', label: 'Hipertensión', icon: <HeartPulse className="me-2" />, variant: 'danger' },
    { key: 'cardiovascular', label: 'Cardiovascular', icon: <Heart className="me-2" />, variant: 'info' },
];

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
            { label: 'Presión Arterial', valor: `${Math.round(d.blood_pressure)} mmHg`, interp: presionBadge(d.blood_pressure) },
            { label: 'Circunferencia de Cintura', valor: `${Math.round(d.waist_circumference)} cm`, interp: <span className="text-muted">Adiposidad central</span> },
            { label: 'Peso', valor: `${Math.round(d.weight)} kg`, interp: <span className="text-muted">Antropometría</span> },
            { label: 'Glucosa', valor: `${Math.round(d.glucose)} mg/dL`, interp: glucosaBadge(d.glucose) },
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

const Educacion = () => {
    const [syntheticData, setSyntheticData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [disease, setDisease] = useState('diabetes');

    const generateDemo = async (target) => {
        const dis = target || disease;
        setDisease(dis);
        setLoading(true);
        setSyntheticData(null);
        try {
            const { data } = await getSyntheticCase(dis);
            setSyntheticData({ ...data, _disease: dis });
        } catch (error) {
            console.error("Error generando caso:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container className="py-5">
            {/* ==============================================
                SECCIÓN 1: ENCICLOPEDIA MÉDICA (Requisito Tutor)
               ============================================== */}
            <div className="mb-5">
                <div className="ps-sec-head mx-auto text-center" style={{ maxWidth: '60ch' }}>
                    <span className="ps-eyebrow">Educación</span>
                    <h2>Biblioteca de enfermedades crónicas</h2>
                    <p>Entendiendo las patologías que afectan a la población general y sus factores de riesgo modificables.</p>
                </div>

                <Tab.Container id="medical-info-tabs" defaultActiveKey="diabetes">
                    <Row>
                        <Col sm={3} className="mb-4">
                            <Nav variant="pills" className="flex-column shadow-sm p-3 bg-white rounded">
                                <Nav.Item>
                                    <Nav.Link eventKey="diabetes"><Droplet className="me-2" />Diabetes T2</Nav.Link>
                                </Nav.Item>
                                <Nav.Item>
                                    <Nav.Link eventKey="hipertension"><HeartPulse className="me-2" />Hipertensión</Nav.Link>
                                </Nav.Item>
                                <Nav.Item>
                                    <Nav.Link eventKey="cardiovascular"><Heart className="me-2" />Cardiovascular</Nav.Link>
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
                                                            <h6 className="text-danger fw-bold"><GraphDownArrow className="me-2" />Impacto en Ecuador</h6>
                                                            <ul className="small mb-0 ps-3">
                                                                <li><strong>Mortalidad:</strong> {info.mortalidad}</li>
                                                                <li className="mt-2"><strong>Población Afectada:</strong> {info.poblacion}</li>
                                                            </ul>
                                                        </div>
                                                    </Col>
                                                    <Col md={6}>
                                                        <div className="p-3 bg-light rounded h-100">
                                                            <h6 className="text-primary fw-bold"><Cpu className="me-2" />Variables Clave (IA)</h6>
                                                            <p className="small mb-0">{info.variables_ia}</p>
                                                        </div>
                                                    </Col>
                                                </Row>

                                                <Row>
                                                    <Col md={6}>
                                                        <h6><ExclamationOctagon className="me-2" />Factores de Riesgo / Hábitos</h6>
                                                        <ListGroup variant="flush" className="small">
                                                            {info.factores.map((f, i) => (
                                                                <ListGroup.Item key={i} className="px-0 py-1">
                                                                    • {f}
                                                                </ListGroup.Item>
                                                            ))}
                                                        </ListGroup>
                                                    </Col>
                                                    <Col md={6}>
                                                        <h6><ExclamationTriangle className="me-2" />Principales Síntomas</h6>
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

            {/* DEMO INTERACTIVA */}
            <div className="bg-light p-5 rounded-3 border">
                <div className="text-center mb-4">
                    <h3><Magic className="me-2" />Laboratorio de Generación (Demo GAN)</h3>
                    <p className="text-muted">
                        Elige una enfermedad y observa cómo la IA "imagina" un paciente con
                        características clínicas coherentes para ese modelo.
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

                    <Button
                        variant="dark"
                        size="lg"
                        onClick={() => generateDemo()}
                        disabled={loading}
                    >
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
                                        <span>Perfil Clínico Generado (IA) · {INFO_ENFERMEDADES[syntheticData._disease]?.titulo}</span>
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
                            </Col>
                        </Row>
                    </div>
                )}
            </div>
        </Container>
    );
};

export default Educacion;