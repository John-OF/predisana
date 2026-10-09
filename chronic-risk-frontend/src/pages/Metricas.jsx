import { useState, useEffect } from 'react';
import { Container, Nav, Table, Row, Col, Spinner, Alert, ButtonGroup, Button } from 'react-bootstrap';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';
import { getMetrics, MODOS, MODO_POR_DEFECTO } from '../services/api';
import { getLabel, getFeatureLabel, grupoOneHot } from '../utils/translations';
import { riskBand, bandNote } from '../utils/riskBand';

const DISEASES = ['diabetes', 'hipertension', 'cardiovascular', 'renal', 'higado'];

const MODEL_LABELS = {
  logistic_regression: 'Regresión Logística',
  logreg: 'Regresión Logística',
  random_forest: 'Random Forest',
  lightgbm: 'LightGBM',
  xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');
const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);
const AUC_MIN_SUBGRUPO = 0.55;  // el del filtro de train_models.py

// Por qué un candidato no pasa el filtro, de lo peor a lo menos malo y con las etiquetas
// de la UI: `motivos` lo guarda el entrenamiento con la clave de cada variable.
const motivosDe = (row) => {
  if (!row.violaciones_signo && !row.auc_subgrupos_cv) return row.motivos || [];
  const signos = Object.entries(row.violaciones_signo || {})
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([f, v]) => `${getFeatureLabel(f)} al revés en el ${pct(v)}`);
  const subgrupos = Object.entries(row.auc_subgrupos_cv || {})
    .filter(([, v]) => v != null && v < AUC_MIN_SUBGRUPO)
    .map(([g, v]) => `AUC ${g.replace(/_/g, ' ')} ${v.toFixed(2)}`);
  return [...signos, ...subgrupos];
};

const Metricas = () => {
  const [selectedDisease, setSelectedDisease] = useState('diabetes');
  // Dos modelos por enfermedad (v2): simplificado y completo.
  const [selectedMode, setSelectedMode] = useState(MODO_POR_DEFECTO);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await getMetrics(selectedDisease, selectedMode);
        setMetrics(data);
      } catch (err) {
        console.error(err);
        setError('No se pudieron cargar las métricas del modelo.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedDisease, selectedMode]);

  const leaderboard = metrics?.leaderboard || [];
  const bestModel = metrics?.best_model;
  const maxAuc = Math.max(...leaderboard.map(r => r.cv_auc_mean || 0), 0.0001);
  const report = metrics?.report;
  const classPos = report?.['1'];
  const calib = metrics?.calibration;
  const bands = metrics?.bands;

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

      <div className="mb-4">
        <p className="text-soft small mb-2">
          Cada enfermedad tiene dos modelos: el <strong>simplificado</strong>, con lo que cualquiera
          sabe de sí mismo (peso y talla autodeclarados, tabaco, diagnósticos previos), y el{' '}
          <strong>completo</strong>, que añade medidas y análisis. Los dos salen de NHANES 2017-2023
          y estiman la enfermedad total: diagnosticada o detectada por análisis o medición.
        </p>
        <ButtonGroup size="sm">
          {MODOS.map(m => (
            <Button key={m} variant={selectedMode === m ? 'primary' : 'outline-primary'} onClick={() => setSelectedMode(m)}>
              {getLabel(m)}
            </Button>
          ))}
        </ButtonGroup>
        {metrics?.objetivo && (
          <p className="text-faint small mt-2 mb-0">
            Objetivo: {metrics.objetivo}. {metrics.n_train?.toLocaleString('es')} personas de entrenamiento,{' '}
            {metrics.n_test?.toLocaleString('es')} de test.
          </p>
        )}
      </div>

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
                <div className="k-sub">de la clasificación del modelo</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">AUC con pesos NHANES</div>
                <div className="k-val">{metrics.auc_test_ponderado != null ? metrics.auc_test_ponderado.toFixed(3) : '—'}</div>
                <div className="k-sub">representativo de adultos de EE. UU.</div>
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
          <p className="text-soft small mb-3">
            Cada algoritmo se ajusta con validación cruzada anidada (la búsqueda de parámetros se
            repite dentro de cada pliegue, así el AUC no está inflado). Y solo puede ganar quien pasa
            el <strong>filtro de validación</strong>: ninguna variable puede mover el riesgo de
            nadie al revés de su sentido clínico (más edad, IMC o presión no pueden bajarlo), y
            tiene que discriminar también por sexo y por edad.
          </p>
          <div className="table-responsive mb-3">
            <Table className="align-middle">
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>AUC (CV anidada)</th>
                  <th>Filtro</th>
                  <th style={{ width: '28%' }}>Comparativa</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => {
                  const isWinner = row.model === bestModel;
                  const w = Math.round(((row.cv_auc_mean || 0) / maxAuc) * 100);
                  const motivos = motivosDe(row);
                  return (
                    <tr key={row.model} className={isWinner ? 'winner' : ''}>
                      <td>
                        {prettyModel(row.model)}
                        {isWinner && <span className="ps-medal">GANADOR</span>}
                      </td>
                      <td className="num">{(row.cv_auc_mean ?? 0).toFixed(4)}</td>
                      <td className="small">
                        {row.pasa_filtro !== false
                          ? <span className="text-success">Pasa</span>
                          : <span className="text-danger" title={motivos.join('; ')}>
                              No pasa: {motivos.slice(0, 2).join('; ')}{motivos.length > 2 ? '…' : ''}
                            </span>}
                      </td>
                      <td>
                        <div className="ps-mini-bar" style={{ width: `${w}%`, opacity: isWinner ? 1 : 0.5 }} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
          </div>
          {(metrics.sin_efecto || []).length > 0 && (
            <Alert variant="secondary" className="small mb-5">
              Las restricciones dejan sin efecto en este modelo: {metrics.sin_efecto.map(f => getFeatureLabel(f)).join(', ')}.
              En datos de un solo momento, quien ya está diagnosticado suele estar en tratamiento (o
              ha dejado de fumar) y no hay señal en el sentido clínico. El simulador lo avisa.
            </Alert>
          )}
          {!(metrics.sin_efecto || []).length && <div className="mb-5" />}

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

          {/* Bandas del simulador: lo que el usuario ve de verdad. La sensibilidad de
              arriba es de la clasificacion si/no del modelo, que el simulador no muestra. */}
          {bands?.test && (
            <div className="mb-5">
              <h3 style={{ fontSize: '1.3rem', marginBottom: '6px' }}>Bandas del simulador (conjunto de test)</h3>
              <p className="text-soft small mb-3">
                El simulador no responde sí o no: muestra la probabilidad calibrada y la lee en tres
                bandas. Así reparten a las personas reales del conjunto de test, y cuántas de cada
                banda tienen de verdad la enfermedad. {bandNote(bands)}
              </p>
              <div className="table-responsive">
                <Table className="align-middle">
                  <thead>
                    <tr>
                      <th>Banda</th>
                      <th>Personas</th>
                      <th>Con la enfermedad</th>
                      <th>Casos que recoge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bands.test.map((b) => {
                      const banda = riskBand({ risk_band: b.band });
                      return (
                        <tr key={b.band}>
                          <td>
                            <span className="ps-risk-pill" style={{ background: banda.bg, color: banda.color }}>{banda.label}</span>
                          </td>
                          <td className="num">{pct(b.share)} <span className="text-faint">({b.n})</span></td>
                          <td className="num">{pct(b.positive_rate)}</td>
                          <td className="num">{pct(b.share_of_positives)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </Table>
              </div>
            </div>
          )}

          {/* Reporte por clase */}
          {report && (
            <>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '6px' }}>Detalle por clase (conjunto de test)</h3>
              <p className="text-soft small mb-3">
                La clasificación del propio modelo: «riesgo» cuando su salida sin calibrar llega a 0,5.
                Es el campo <code>prediction</code> de la API y de aquí sale la sensibilidad de arriba;
                el simulador no la muestra, enseña la probabilidad y su banda.
              </p>
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
                <p className="text-soft small mb-0">
                  {[...new Set(metrics.features.map(f => getLabel(grupoOneHot(f) ?? f)))].join(', ')}.
                </p>
                {(metrics.definitorias || []).length > 0 && (
                  <p className="text-faint small mt-2 mb-0">
                    No entran en el modelo porque definen la enfermedad: {metrics.definitorias.map(f => getLabel(f)).join(', ')}.
                    Si se aportan, las interpreta la guía (ADA, ACC/AHA).
                  </p>
                )}
              </div>
            </>
          )}
        </>
      )}
    </Container>
  );
};

export default Metricas;
