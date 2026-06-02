import { useState, useEffect } from 'react';
import { Container, Card, Nav, Table, Row, Col, Spinner, Alert } from 'react-bootstrap';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getMetrics } from '../services/api';
import { getLabel } from '../utils/translations';

const DISEASES = ['diabetes', 'hipertension', 'obesidad', 'cardiovascular'];

const Metricas = () => {
    const [selectedDisease, setSelectedDisease] = useState('diabetes');
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const { data } = await getMetrics(selectedDisease);
                setMetrics(data);
            } catch (err) {
                console.error(err);
                setError("No se pudieron cargar las métricas del modelo.");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [selectedDisease]);

    // Prepara datos para el gráfico (Precision, Recall, F1)
    const getChartData = () => {
        if (!metrics || !metrics.report) return [];
        // El reporte trae claves "0" (Sano), "1" (Riesgo), "accuracy", etc.
        const class0 = metrics.report["0"];
        const class1 = metrics.report["1"];
        
        return [
            { name: 'Clase 0 (Bajo Riesgo)', precision: class0.precision, recall: class0.recall, f1: class0['f1-score'] },
            { name: 'Clase 1 (Alto Riesgo)', precision: class1.precision, recall: class1.recall, f1: class1['f1-score'] },
        ];
    };

    return (
        <Container className="py-4">
            <h2 className="mb-4">Métricas de Rendimiento de los Modelos</h2>
            
            <Card className="mb-4">
                <Card.Header>
                    <Nav variant="tabs" defaultActiveKey="diabetes">
                        {DISEASES.map(d => (
                            <Nav.Item key={d}>
                                <Nav.Link 
                                    active={selectedDisease === d}
                                    onClick={() => setSelectedDisease(d)}
                                    className="text-capitalize"
                                >
                                    {getLabel(d)}
                                </Nav.Link>
                            </Nav.Item>
                        ))}
                    </Nav>
                </Card.Header>
                <Card.Body>
                    {loading && <div className="text-center py-5"><Spinner animation="border" /></div>}
                    {error && <Alert variant="danger">{error}</Alert>}
                    
                    {metrics && !loading && (
                        <div>
                            <Row className="mb-4">
                                <Col md={4}>
                                    <Card className="text-center bg-light border-0 h-100">
                                        <Card.Body>
                                            <h6>AUC (Área Bajo la Curva)</h6>
                                            <h2 className="text-primary fw-bold">{(metrics.auc * 100).toFixed(1)}%</h2>
                                            <small className="text-muted">Capacidad de distinción global</small>
                                        </Card.Body>
                                    </Card>
                                </Col>
                                <Col md={4}>
                                    <Card className="text-center bg-light border-0 h-100">
                                        <Card.Body>
                                            <h6>Exactitud (Accuracy)</h6>
                                            <h2 className="text-success fw-bold">
                                                {(metrics.report.accuracy * 100).toFixed(1)}%
                                            </h2>
                                            <small className="text-muted">Predicciones correctas totales</small>
                                        </Card.Body>
                                    </Card>
                                </Col>
                                <Col md={4}>
                                    <Card className="text-center bg-light border-0 h-100">
                                        <Card.Body>
                                            <h6>Features Utilizados</h6>
                                            <h2 className="text-info fw-bold">{metrics.features ? metrics.features.length : 0}</h2>
                                            <small className="text-muted">Variables de entrada</small>
                                        </Card.Body>
                                    </Card>
                                </Col>
                            </Row>

                            <h5 className="mb-3">Detalle por Clase (Reporte de Clasificación)</h5>
                            <div style={{ height: 300, width: '100%' }}>
                                <ResponsiveContainer>
                                    <BarChart data={getChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" />
                                        <YAxis domain={[0, 1]} />
                                        <Tooltip />
                                        <Legend />
                                        <Bar dataKey="precision" name="Precisión" fill="#8884d8" />
                                        <Bar dataKey="recall" name="Sensibilidad (Recall)" fill="#82ca9d" />
                                        <Bar dataKey="f1" name="F1-Score" fill="#ffc658" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="mt-4">
                                <h6>Variables utilizadas en este modelo:</h6>
                                <p className="text-muted small">
                                    {metrics.features.map(f => getLabel(f)).join(', ')}.
                                </p>
                            </div>
                        </div>
                    )}
                </Card.Body>
            </Card>
        </Container>
    );
};

export default Metricas;