import { Container, Card, Alert } from 'react-bootstrap';

const Aviso = () => {
    return (
        <Container className="py-5">
            <h2 className="mb-4">Aviso Legal y Limitación de Responsabilidad</h2>
            
            <Alert variant="warning" className="mb-4">
                <Alert.Heading>⚠ Importante: Herramienta No Clínica</Alert.Heading>
                <p>
                    Este sistema es un prototipo académico desarrollado como parte de un proyecto de tesis de Ingeniería de Software.
                    <strong> NO es un dispositivo médico ni una herramienta de diagnóstico clínico.</strong>
                </p>
            </Alert>

            <Card className="shadow-sm mb-4">
                <Card.Header as="h5">Propósito del Proyecto</Card.Header>
                <Card.Body>
                    <Card.Text>
                        El objetivo de este portal es estrictamente <strong>educativo y demostrativo</strong>. 
                        Busca ilustrar cómo funcionan los algoritmos de Inteligencia Artificial (específicamente Regresión Logística y Redes Generativas Antagónicas) en el contexto de la salud.
                    </Card.Text>
                </Card.Body>
            </Card>

            <Card className="shadow-sm mb-4">
                <Card.Header as="h5">Uso de Datos</Card.Header>
                <Card.Body>
                    <ul>
                        <li>Los modelos han sido entrenados con datasets públicos y datos sintéticos.</li>
                        <li>Las predicciones son estimaciones estadísticas y pueden contener márgenes de error.</li>
                        <li><strong>Privacidad:</strong> Este sitio no almacena datos personales identificables (PII) de los usuarios que utilizan el simulador.</li>
                    </ul>
                </Card.Body>
            </Card>

            <Card className="shadow-sm border-danger">
                <Card.Header as="h5" className="text-danger">Exención de Responsabilidad Médica</Card.Header>
                <Card.Body>
                    <Card.Text>
                        La información proporcionada por este sistema no debe utilizarse para tomar decisiones sobre tratamientos médicos, medicación o cambios en el estilo de vida sin consultar previamente a un profesional de la salud certificado.
                        <br /><br />
                        Si usted presenta síntomas o preocupaciones sobre su salud, acuda inmediatamente a un médico o centro de salud.
                    </Card.Text>
                </Card.Body>
            </Card>
        </Container>
    );
};

export default Aviso;