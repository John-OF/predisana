import { useState, useEffect } from 'react';
import { Container, Nav, Table, Row, Col, Spinner, Alert, ButtonGroup, Button } from 'react-bootstrap';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';
import { getMetrics } from '../services/api';
import { getLabel } from '../utils/translations';

const DISEASES = ['diabetes', 'hipertension', 'cardiovascular'];

const MODEL_LABELS = {
  logistic_regression: 'Regresión Logística',
  logreg: 'Regresión Logística',
  random_forest: 'Random Forest',
  lightgbm: 'LightGBM',
  xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');
const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);

const Metricas = () => {
  const [selectedDisease, setSelectedDisease] = useState('diabetes');
  // Diabetes tiene un modelo HÍBRIDO: la variante con glucosa se pide como
  // 'diabetes_glucosa'. Para el resto de enfermedades el toggle no aplica.
  const [showGlucose, setShowGlucose] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const metricsKey = selectedDisease === 'diabetes' && showGlucose ? 'diabetes_glucosa' : selectedDisease;

  const selectDisease = (d) => {
    setShowGlucose(false);      // el toggle de glucosa solo vive en diabetes
    setSelectedDisease(d);
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await getMetrics(metricsKey);
        setMetrics(data);
      } catch (err) {
        console.error(err);
        setError('No se pudieron cargar las métricas del modelo.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [metricsKey]);

  const leaderboard = metrics?.leaderboard || [];
  const bestModel = metrics?.best_model;
  const maxAuc = Math.max(...leaderboard.map(r => r.cv_auc_mean || 0), 0.0001);
  const report = metrics?.report;
  const classPos = report?.['1'];
  const calib = metrics?.calibration;

  return (
    <Container className="py-5">
      <div className="ps-sec-head">
        <span className="ps-eyebrow">Métricas</span>
        <h2>Rendimiento de los modelos</h2>
        <p>Comparativa honesta de algoritmos. La probabilidad servida corresponde exactamente a este AUC (sin reglas que la inflen).</p>
      </div>

      <Nav variant="tabs" className="mb-4">
        {DISEASES.map(d => (
          <Nav.Item key={d}>
            <Nav.Link active={selectedDisease === d} onClick={() => selectDisease(d)} className="text-capitalize px-4">
              {getLabel(d)}
            </Nav.Link>
          </Nav.Item>
        ))}
      </Nav>

      {selectedDisease === 'diabetes' && (
        <div className="mb-4">
          <p className="text-soft small mb-2">
            La diabetes usa un modelo <strong>híbrido</strong>: uno con solo datos que cualquiera
            puede responder, y una variante que añade la <strong>glucosa sérica</strong> (opcional)
            cuando la persona la conoce. Compara ambos:
          </p>
          <ButtonGroup size="sm">
            <Button variant={!showGlucose ? 'primary' : 'outline-primary'} onClick={() => setShowGlucose(false)}>
              Sin glucosa · respondible
            </Button>
            <Button variant={showGlucose ? 'primary' : 'outline-primary'} onClick={() => setShowGlucose(true)}>
              Con glucosa · híbrido
            </Button>
          </ButtonGroup>
        </div>
      )}

      {loading && <div className="text-center py-5"><Spinner animation="border" variant="primary" /></div>}
      {error && <Alert variant="danger">{error}</Alert>}

      {metrics && !loading && (
        <>
          {/* KPIs */}
          <Row className="g-3 mb-4">
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">AUC (test)</div>
                <div className="k-val">{(metrics.auc ?? metrics.auc_test ?? 0).toFixed(3)}</div>
                <div className="k-sub">{prettyModel(bestModel)}</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Sensibilidad</div>
                <div className="k-val">{classPos ? pct(classPos.recall) : '—'}</div>
                <div className="k-sub">recall · clase riesgo</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Exactitud</div>
                <div className="k-val">{report ? pct(report.accuracy) : '—'}</div>
                <div className="k-sub">accuracy global</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Variables</div>
                <div className="k-val">{metrics.features ? metrics.features.length : 0}</div>
                <div className="k-sub">features de entrada</div>
              </div>
            </Col>
          </Row>

          {/* Leaderboard */}
          <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Leaderboard de algoritmos</h3>
          <p className="text-soft small mb-3">Selección por AUC en validación cruzada (5-fold) sobre el conjunto de entrenamiento.</p>
          <div className="table-responsive mb-5">
            <Table className="align-middle">
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>AUC (CV)</th>
                  <th>± Desv.</th>
                  <th style={{ width: '32%' }}>Comparativa</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => {
                  const isWinner = row.model === bestModel;
                  const w = Math.round(((row.cv_auc_mean || 0) / maxAuc) * 100);
                  return (
                    <tr key={row.model} className={isWinner ? 'winner' : ''}>
                      <td>
                        {prettyModel(row.model)}
                        {isWinner && <span className="ps-medal">GANADOR</span>}
                      </td>
                      <td className="num">{(row.cv_auc_mean ?? 0).toFixed(4)}</td>
                      <td className="num text-faint">±{(row.cv_auc_std ?? 0).toFixed(4)}</td>
                      <td>
                        <div className="ps-mini-bar" style={{ width: `${w}%`, opacity: isWinner ? 1 : 0.5 }} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
          </div>

          {/* Curva de calibración (fiabilidad) */}
          {calib && (
            <div className="mb-5">
              <h3 style={{ fontSize: '1.3rem', marginBottom: '6px' }}>Calibración de probabilidades</h3>
              <p className="text-soft small mb-3">
                El AUC solo mide el <em>orden</em> de los scores. La curva de fiabilidad comprueba
                que un "30%" signifique de verdad "30% de los casos así son positivos": los puntos
                deben caer sobre la diagonal. Aplicamos una regresión isotónica que reescala la
                salida del modelo sin alterar su ranking (AUC intacto).
              </p>
              <Row className="g-3">
                <Col lg={7}>
                  <div style={{ width: '100%', height: 320 }}>
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.2)" />
                        <XAxis type="number" dataKey="mean_pred" name="Prob. predicha" domain={[0, 1]}
                          tickFormatter={(v) => v.toFixed(1)} fontSize={12}
                          label={{ value: 'Probabilidad predicha', position: 'insideBottom', offset: -15, fontSize: 12 }} />
                        <YAxis type="number" dataKey="frac_pos" name="Frac. positivos" domain={[0, 1]}
                          tickFormatter={(v) => v.toFixed(1)} fontSize={12}
                          label={{ value: 'Fracción real de positivos', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                        <ZAxis range={[60, 60]} />
                        <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)}
                          cursor={{ strokeDasharray: '3 3' }} />
                        <Legend verticalAlign="top" height={30} />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                          stroke="rgba(128,128,128,0.55)" strokeDasharray="5 5" ifOverflow="extendDomain" />
                        <Scatter name="Sin calibrar" data={calib.raw_curve} fill="#c98a2b" line lineType="joint" />
                        <Scatter name="Calibrado" data={calib.calibrated_curve} fill="#2f9e8f" line lineType="joint" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </Col>
                <Col lg={5}>
                  <div className="d-flex flex-column gap-3 h-100 justify-content-center">
                    <div className="ps-kpi">
                      <div className="k-lbl">Brier score · sin calibrar</div>
                      <div className="k-val">{calib.brier_raw?.toFixed(4)}</div>
                      <div className="k-sub">error cuadrático medio de probabilidad (menor = mejor)</div>
                    </div>
                    <div className="ps-kpi" style={{ borderColor: 'var(--accent, #2f9e8f)' }}>
                      <div className="k-lbl">Brier score · calibrado</div>
                      <div className="k-val">{calib.brier_calibrated?.toFixed(4)}</div>
                      <div className="k-sub">
                        {calib.brier_raw > 0
                          ? `mejora del ${(((calib.brier_raw - calib.brier_calibrated) / calib.brier_raw) * 100).toFixed(0)}%`
                          : 'isotónica sobre predicciones out-of-fold'}
                      </div>
                    </div>
                  </div>
                </Col>
              </Row>
            </div>
          )}

          {/* Reporte por clase */}
          {report && (
            <>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Detalle por clase (conjunto de test)</h3>
              <div className="table-responsive">
                <Table className="align-middle">
                  <thead>
                    <tr>
                      <th>Clase</th>
                      <th>Precisión</th>
                      <th>Sensibilidad</th>
                      <th>F1</th>
                      <th>Soporte</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['0', '1'].map((c) => report[c] && (
                      <tr key={c}>
                        <td>{c === '1' ? 'Riesgo (positiva)' : 'Bajo riesgo (negativa)'}</td>
                        <td className="num">{pct(report[c].precision)}</td>
                        <td className="num">{pct(report[c].recall)}</td>
                        <td className="num">{pct(report[c]['f1-score'])}</td>
                        <td className="num text-faint">{Math.round(report[c].support)}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>

              <div className="mt-4">
                <h5 style={{ fontSize: '1.05rem' }}>Variables del modelo</h5>
                <p className="text-soft small mb-0">{metrics.features.map(f => getLabel(f)).join(', ')}.</p>
              </div>
            </>
          )}
        </>
      )}
    </Container>
  );
};

export default Metricas;
