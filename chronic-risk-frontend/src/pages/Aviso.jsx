import { Container, Card, Alert } from 'react-bootstrap';
import { ExclamationTriangleFill } from 'react-bootstrap-icons';

const Aviso = () => {
    return (
        <Container className="py-5">
            <div className="ps-sec-head">
                <span className="ps-eyebrow">Aviso legal</span>
                <h2>Aviso legal y limitación de responsabilidad</h2>
            </div>

            <Alert variant="warning" className="mb-4">
                <Alert.Heading><ExclamationTriangleFill className="me-2" />Importante: Herramienta No Clínica</Alert.Heading>
                <p>
                    Este sistema es un prototipo de software de portafolio con fines educativos y demostrativos.
                    <strong> NO es un dispositivo médico ni una herramienta de diagnóstico clínico.</strong>
                </p>
            </Alert>

            <Card className="shadow-sm mb-4">
                <Card.Header as="h5">Propósito del Proyecto</Card.Header>
                <Card.Body>
                    <Card.Text>
                        El objetivo de este portal es estrictamente <strong>educativo y demostrativo</strong>. 
                        Busca ilustrar cómo funcionan los algoritmos de Inteligencia Artificial (un stack multi-modelo —Regresión Logística, Random Forest y LightGBM— con explicabilidad SHAP y datos sintéticos CTGAN) en el contexto de la salud.
                    </Card.Text>
                </Card.Body>
            </Card>

            <Card className="shadow-sm mb-4">
                <Card.Header as="h5">Uso de Datos</Card.Header>
                <Card.Body>
                    <ul>
                        <li>Los modelos han sido entrenados con datasets públicos y datos sintéticos.</li>
                        <li>Las predicciones son estimaciones estadísticas y pueden contener márgenes de error.</li>
                        <li><strong>Privacidad:</strong> Las simulaciones se registran de forma <strong>anónima</strong> (los datos clínicos ingresados, sin ningún dato personal identificable / PII) con fines estadísticos y de mejora del sistema. No se solicita ni se almacena información que permita identificarte.</li>
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