import { useState, useEffect } from 'react';
import { Container, Nav, Table, Row, Col, Spinner, Alert } from 'react-bootstrap';
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
        setError('No se pudieron cargar las métricas del modelo.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedDisease]);

  const leaderboard = metrics?.leaderboard || [];
  const bestModel = metrics?.best_model;
  const maxAuc = Math.max(...leaderboard.map(r => r.cv_auc_mean || 0), 0.0001);
  const report = metrics?.report;
  const classPos = report?.['1'];

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
            <Nav.Link active={selectedDisease === d} onClick={() => setSelectedDisease(d)} className="text-capitalize px-4">
              {getLabel(d)}
            </Nav.Link>
          </Nav.Item>
        ))}
      </Nav>

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
