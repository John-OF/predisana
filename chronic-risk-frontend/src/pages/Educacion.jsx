import { Container, Row, Col, Card, Badge, Tab, Nav, ListGroup } from 'react-bootstrap';
import { Droplet, HeartPulse, Heart, Funnel, GraphDownArrow, Cpu, ExclamationOctagon, ExclamationTriangle } from 'react-bootstrap-icons';

// `variables_ia` describe los modelos servidos (v2: simplificado y completo), no la
// medicina en general: el orden sale de la media de |SHAP| sobre el test real.
// tests/test_textos_educacion.py (backend) comprueba cada afirmacion contra los
// modelos: si un reentrenamiento cambia el orden, falla y hay que revisar el texto.
const INFO_ENFERMEDADES = {
    diabetes: {
        titulo: "Diabetes Tipo 2",
        color: "primary",
        definicion: "Trastorno metabólico crónico caracterizado por niveles elevados de glucosa en sangre, debido a que el cuerpo no produce suficiente insulina o no la utiliza eficazmente.",
        mortalidad: "La diabetes mellitus es una de las principales causas de muerte no violenta a nivel mundial.",
        poblacion: "Se estima que afecta a cerca del 10% de la población adulta mundial (más de 500 millones de personas).",
        organos: ["Páncreas", "Riñones (Nefropatía)", "Ojos (Retina)", "Corazón", "Nervios periféricos"],
        variables_ia: "Con lo que cualquiera sabe de sí mismo, lo que más pesa es la edad, seguida del IMC y de tener la presión alta. En el modo completo cuentan además la albúmina en orina, la cintura y el colesterol HDL. La HbA1c y la glucosa no entran en el modelo: definen la diabetes, así que se interpretan aparte con los umbrales de la ADA.",
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
        poblacion: "Afecta aproximadamente a 1 de cada 3 adultos en el mundo (cerca del 30%).",
        organos: ["Corazón (Insuficiencia)", "Arterias", "Cerebro", "Riñones"],
        variables_ia: "Lo que más pesa es la edad, seguida del IMC y del colesterol alto. En el modo completo cuentan además la albúmina en orina y la cintura. La presión no entra en el modelo, porque es la medida con la que se diagnostica la hipertensión: si la aportas, se interpreta aparte con los umbrales ACC/AHA.",
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
        titulo: "Enfermedad Cardiovascular",
        color: "info",
        definicion: "Grupo de desórdenes del corazón y los vasos sanguíneos, incluyendo cardiopatías coronarias y enfermedades cerebrovasculares.",
        mortalidad: "Es la causa #1 de muerte a nivel mundial (aprox. 32% de las defunciones).",
        poblacion: "El riesgo aumenta significativamente en hombres >45 años y mujeres >55 años.",
        organos: ["Corazón", "Cerebro", "Sistema circulatorio completo"],
        variables_ia: "Lo que más pesa es la edad, seguida de tener la presión alta y del tabaco (fumar o haber fumado). En el modo completo, la presión medida, el colesterol total y la HbA1c no cambian la estimación: quien ya tuvo un evento suele estar en tratamiento, así que en estos datos no dan señal. El simulador lo avisa.",
        factores: [
            "Tabaquismo (Factor crítico).",
            "Colesterol LDL alto.",
            "Diabetes e Hipertensión no controladas.",
            "Inactividad física."
        ],
        sintomas: ["Dolor u opresión en el pecho (Angina)", "Falta de aire", "Entumecimiento en extremidades", "Palpitaciones"],
        tratamiento: "Prevención primaria (estilo de vida) y secundaria (fármacos como estatinas, aspirina, cirugías)."
    },
    renal: {
        titulo: "Enfermedad Renal Crónica",
        color: "success",
        definicion: "Daño en los riñones o pérdida de su capacidad de filtrar la sangre que dura más de 3 meses. Se detecta con dos análisis sencillos: el filtrado glomerular (eGFR, en sangre) por debajo de 60 o la albúmina en orina desde 30 mg/g.",
        mortalidad: "Causa más de un millón de muertes al año en el mundo y multiplica el riesgo de infarto e ictus.",
        poblacion: "Afecta a más del 10% de los adultos del mundo y casi nadie lo sabe: en estos datos, solo el 19% de quienes la tienen estaba diagnosticado.",
        organos: ["Riñones", "Corazón y vasos", "Huesos", "Sangre (Anemia)"],
        variables_ia: "Lo que más pesa es la edad, seguida de tener la presión alta y la diabetes, sus dos primeras causas. En el modo completo cuentan además la HbA1c y la presión sistólica medida. El filtrado glomerular y la albúmina en orina no entran en el modelo: definen la enfermedad, así que se interpretan aparte con los umbrales de KDIGO. El colesterol alto diagnosticado no cambia la estimación: quien lo tiene suele tomar estatinas, y en estos datos no da señal.",
        factores: [
            "Diabetes (primera causa).",
            "Hipertensión arterial (segunda causa).",
            "Enfermedad cardiovascular y obesidad.",
            "Edad avanzada y antecedentes familiares."
        ],
        sintomas: ["Generalmente asintomática hasta fases avanzadas.", "Hinchazón de piernas y tobillos.", "Cansancio y falta de apetito.", "Orina espumosa."],
        tratamiento: "No tiene cura, pero se puede frenar: controlar la presión y la glucosa, fármacos que protegen el riñón y evitar los que lo dañan. En la fase final, diálisis o trasplante."
    }
};

const Educacion = () => {
    return (
        <Container className="py-5">
            {/* ==============================================
                ENCICLOPEDIA MÉDICA — Biblioteca de enfermedades
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
                                <Nav.Item>
                                    <Nav.Link eventKey="renal"><Funnel className="me-2" />Renal</Nav.Link>
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
                                                            <h6 className="text-danger fw-bold"><GraphDownArrow className="me-2" />Impacto epidemiológico</h6>
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
        </Container>
    );
};

export default Educacion;
