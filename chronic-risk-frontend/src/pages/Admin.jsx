import { useState, useEffect, useCallback } from 'react';
import { Container, Row, Col, Form, Button, Table, Spinner, Alert, Badge } from 'react-bootstrap';
import { ShieldLock, BoxArrowRight, ArrowClockwise, Download, FunnelFill } from 'react-bootstrap-icons';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell,
} from 'recharts';
import {
  verifyAdmin, getAdminStats, getAdminPredictions, downloadAdminCsv,
} from '../services/api';
import { getLabel } from '../utils/translations';

// El token vive en sessionStorage: se borra al cerrar la pestaña (más seguro que
// localStorage para una credencial de admin). NO es auth de usuario.
const TOKEN_KEY = 'predisana_admin_token';

const MODEL_LABELS = {
  logistic_regression: 'Regresión Logística', logreg: 'Regresión Logística',
  random_forest: 'Random Forest', lightgbm: 'LightGBM', xgboost: 'XGBoost',
};
const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');
const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);
const fmtTs = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

const Admin = () => {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) || '');
  const [authed, setAuthed] = useState(false);
  const [pwd, setPwd] = useState('');
  const [authError, setAuthError] = useState(null);
  const [checking, setChecking] = useState(false);

  const [stats, setStats] = useState(null);
  const [preds, setPreds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataError, setDataError] = useState(null);

  // Filtro por rango de fechas (YYYY-MM-DD, ambos opcionales).
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [histDisease, setHistDisease] = useState('');
  const [exporting, setExporting] = useState(false);

  const loadDashboard = useCallback(async (tok, range = {}) => {
    setLoading(true);
    setDataError(null);
    try {
      const [s, p] = await Promise.all([
        getAdminStats(tok, range),
        getAdminPredictions(tok, { limit: 50, ...range }),
      ]);
      setStats(s.data);
      setPreds(p.data.items || []);
    } catch (err) {
      console.error(err);
      if (err?.response?.status === 401) {
        // Token dejó de ser válido: forzar re-login.
        handleLogout();
        setAuthError('La sesión expiró o el token es inválido.');
      } else if (err?.response?.status === 503) {
        setDataError('El panel admin está deshabilitado en el servidor (falta configurar ADMIN_TOKEN).');
      } else {
        setDataError('No se pudieron cargar los datos del panel.');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Si ya hay token guardado, intenta entrar directo al montar.
  useEffect(() => {
    if (!token) return;
    let active = true;
    (async () => {
      try {
        await verifyAdmin(token);
        if (active) { setAuthed(true); loadDashboard(token); }
      } catch {
        if (active) { sessionStorage.removeItem(TOKEN_KEY); setToken(''); }
      }
    })();
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setChecking(true);
    setAuthError(null);
    try {
      await verifyAdmin(pwd);
      sessionStorage.setItem(TOKEN_KEY, pwd);
      setToken(pwd);
      setAuthed(true);
      setPwd('');
      loadDashboard(pwd);
    } catch (err) {
      if (err?.response?.status === 503) {
        setAuthError('El panel admin está deshabilitado en el servidor (falta configurar ADMIN_TOKEN).');
      } else {
        setAuthError('Token incorrecto.');
      }
    } finally {
      setChecking(false);
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem(TOKEN_KEY);
    setToken('');
    setAuthed(false);
    setStats(null);
    setPreds([]);
  };

  const currentRange = () => ({ from: dateFrom || undefined, to: dateTo || undefined });

  const applyFilter = () => loadDashboard(token, currentRange());
  const clearFilter = () => { setDateFrom(''); setDateTo(''); loadDashboard(token, {}); };

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadAdminCsv(token, currentRange());
    } catch (err) {
      console.error(err);
      setDataError('No se pudo exportar el CSV.');
    } finally {
      setExporting(false);
    }
  };

  // ---- Gate de login ----
  if (!authed) {
    return (
      <Container className="py-5" style={{ maxWidth: 460 }}>
        <div className="ps-sec-head text-center">
          <span className="ps-eyebrow">Panel privado</span>
          <h2><ShieldLock className="me-2" />Acceso de administrador</h2>
          <p>Área dev-only. Los usuarios nunca se loguean; este panel solo muestra analítica de uso anónima.</p>
        </div>
        <Form onSubmit={handleLogin} className="ps-card p-4">
          <Form.Group className="ps-field mb-3">
            <Form.Label>Token de administrador</Form.Label>
            <Form.Control
              type="password"
              value={pwd}
              onChange={(e) => setPwd(e.target.value)}
              placeholder="X-Admin-Token"
              autoFocus
            />
          </Form.Group>
          {authError && <Alert variant="danger" className="py-2 small">{authError}</Alert>}
          <Button type="submit" variant="primary" className="w-100 fw-bold" disabled={checking || !pwd}>
            {checking ? <Spinner size="sm" animation="border" /> : 'Entrar'}
          </Button>
        </Form>
      </Container>
    );
  }

  // ---- Dashboard ----
  const byDisease = stats?.by_disease || [];
  const timeline = stats?.timeline || [];
  const maxCount = Math.max(...byDisease.map(d => d.count || 0), 1);

  // Uso por hora del día (0-23).
  const hourlyData = (stats?.hourly || []).map((c, h) => ({
    hour: String(h).padStart(2, '0'), count: c,
  }));

  // Histograma de probabilidad para la enfermedad elegida (10 bins 0..1).
  const nBins = stats?.prob_bins || 10;
  const diseasesWithData = byDisease.filter(d => d.count > 0).map(d => d.disease);
  const activeHistDisease = histDisease || diseasesWithData[0] || (byDisease[0]?.disease);
  const histBins = (stats?.prob_histogram?.[activeHistDisease] || []).map((c, i) => ({
    band: `${Math.round((i / nBins) * 100)}–${Math.round(((i + 1) / nBins) * 100)}%`,
    count: c,
    high: i >= nBins / 2,
  }));

  // Top features SHAP más frecuentes.
  const topFeatures = stats?.top_features || [];
  const maxFeatCount = Math.max(...topFeatures.map(f => f.count || 0), 1);

  return (
    <Container className="py-5">
      <div className="d-flex justify-content-between align-items-start flex-wrap gap-2 mb-4">
        <div className="ps-sec-head mb-0">
          <span className="ps-eyebrow">Panel privado</span>
          <h2>Analítica de uso</h2>
          <p className="mb-0">Agregada y anónima. Sin datos personales: solo inputs de salud y la salida del modelo.</p>
        </div>
        <div className="d-flex gap-2">
          <Button variant="outline-primary" size="sm" onClick={() => loadDashboard(token, currentRange())} disabled={loading}>
            <ArrowClockwise className="me-1" />Refrescar
          </Button>
          <Button variant="outline-secondary" size="sm" onClick={handleLogout}>
            <BoxArrowRight className="me-1" />Salir
          </Button>
        </div>
      </div>

      {/* Barra de filtro por fechas + export */}
      <div className="ps-card p-3 mb-4">
        <div className="d-flex align-items-end gap-3 flex-wrap">
          <Form.Group>
            <Form.Label className="small text-faint mb-1">Desde</Form.Label>
            <Form.Control type="date" size="sm" value={dateFrom} max={dateTo || undefined}
              onChange={(e) => setDateFrom(e.target.value)} />
          </Form.Group>
          <Form.Group>
            <Form.Label className="small text-faint mb-1">Hasta</Form.Label>
            <Form.Control type="date" size="sm" value={dateTo} min={dateFrom || undefined}
              onChange={(e) => setDateTo(e.target.value)} />
          </Form.Group>
          <Button variant="primary" size="sm" onClick={applyFilter} disabled={loading}>
            <FunnelFill className="me-1" />Aplicar
          </Button>
          {(dateFrom || dateTo) && (
            <Button variant="outline-secondary" size="sm" onClick={clearFilter} disabled={loading}>
              Limpiar
            </Button>
          )}
          <div className="ms-auto">
            <Button variant="outline-primary" size="sm" onClick={handleExport} disabled={exporting || loading}>
              {exporting ? <Spinner size="sm" animation="border" /> : <><Download className="me-1" />Exportar CSV</>}
            </Button>
          </div>
        </div>
      </div>

      {loading && <div className="text-center py-5"><Spinner animation="border" variant="primary" /></div>}
      {dataError && <Alert variant="danger">{dataError}</Alert>}

      {stats && !loading && (
        <>
          {/* KPIs */}
          <Row className="g-3 mb-4">
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Simulaciones</div>
                <div className="k-val">{stats.total}</div>
                <div className="k-sub">predicciones registradas</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Sesiones</div>
                <div className="k-val">{stats.distinct_sessions}</div>
                <div className="k-sub">visitantes anónimos únicos</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Enfermedades</div>
                <div className="k-val">{byDisease.length}</div>
                <div className="k-sub">modelos consultados</div>
              </div>
            </Col>
            <Col sm={6} lg={3}>
              <div className="ps-kpi">
                <div className="k-lbl">Días activos</div>
                <div className="k-val">{timeline.length}</div>
                <div className="k-sub">con al menos una simulación</div>
              </div>
            </Col>
          </Row>

          {/* Timeline */}
          {timeline.length > 0 && (
            <div className="mb-5">
              <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Simulaciones por día</h3>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={timeline} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Area type="monotone" dataKey="count" name="Simulaciones" stroke="#0e7c7b" fill="#0e7c7b" fillOpacity={0.18} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Distribución de probabilidades + uso por hora */}
          <Row className="g-4 mb-5">
            <Col lg={6}>
              <div className="d-flex justify-content-between align-items-center mb-2 flex-wrap gap-2">
                <h3 style={{ fontSize: '1.3rem', margin: 0 }}>Distribución de probabilidades</h3>
                {diseasesWithData.length > 1 && (
                  <Form.Select size="sm" style={{ width: 'auto' }}
                    value={activeHistDisease}
                    onChange={(e) => setHistDisease(e.target.value)}>
                    {byDisease.map(d => (
                      <option key={d.disease} value={d.disease}>{getLabel(d.disease)}</option>
                    ))}
                  </Form.Select>
                )}
              </div>
              <p className="text-soft small mb-2">Cuántas simulaciones caen en cada franja de riesgo (probabilidad calibrada) para {getLabel(activeHistDisease)}.</p>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={histBins} margin={{ top: 6, right: 12, left: -12, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="band" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={50} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Simulaciones" radius={[4, 4, 0, 0]}>
                    {histBins.map((b, i) => (
                      <Cell key={i} fill={b.high ? '#c8736a' : '#2f9e8f'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Col>
            <Col lg={6}>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '8px' }}>Simulaciones por hora del día</h3>
              <p className="text-soft small mb-2">Cuándo se usa el simulador (hora del servidor, 0–23).</p>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={hourlyData} margin={{ top: 6, right: 12, left: -12, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="hour" tick={{ fontSize: 10 }} interval={1} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Simulaciones" fill="#0e7c7b" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Col>
          </Row>

          {/* Top features SHAP */}
          {topFeatures.length > 0 && (
            <div className="mb-5">
              <h3 style={{ fontSize: '1.3rem', marginBottom: '6px' }}>Factores de riesgo más frecuentes</h3>
              <p className="text-soft small mb-3">Variables que más veces aparecen entre los 5 factores SHAP de mayor peso, sobre todas las simulaciones del rango.</p>
              <div className="table-responsive">
                <Table className="align-middle">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Veces en el top</th>
                      <th>Impacto medio</th>
                      <th style={{ width: '38%' }}>Frecuencia</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topFeatures.map((f) => (
                      <tr key={f.feature}>
                        <td>{getLabel(f.feature)}</td>
                        <td className="num">{f.count}</td>
                        <td className="num text-faint">{f.avg_abs_shap?.toFixed(3)}</td>
                        <td><div className="ps-mini-bar" style={{ width: `${Math.round((f.count / maxFeatCount) * 100)}%` }} /></td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            </div>
          )}

          {/* Por enfermedad */}
          <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Uso por enfermedad</h3>
          <div className="table-responsive mb-5">
            <Table className="align-middle">
              <thead>
                <tr>
                  <th>Enfermedad</th>
                  <th>Modelo</th>
                  <th>Simulaciones</th>
                  <th>Tasa positiva</th>
                  <th>Prob. media</th>
                  <th style={{ width: '24%' }}>Volumen</th>
                </tr>
              </thead>
              <tbody>
                {byDisease.map((d) => (
                  <tr key={d.disease}>
                    <td className="text-capitalize">{getLabel(d.disease)}</td>
                    <td>{prettyModel(d.model)}</td>
                    <td className="num">{d.count}</td>
                    <td className="num">{pct(d.positive_rate)}</td>
                    <td className="num text-faint">{pct(d.avg_probability)}</td>
                    <td><div className="ps-mini-bar" style={{ width: `${Math.round((d.count / maxCount) * 100)}%` }} /></td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>

          {/* Predicciones recientes */}
          <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Simulaciones recientes</h3>
          <p className="text-soft small mb-3">Últimas {preds.length} (máx. 50). Anónimas: el <code>session_id</code> es un UUID aleatorio sin PII.</p>
          <div className="table-responsive">
            <Table className="align-middle" size="sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Fecha</th>
                  <th>Enfermedad</th>
                  <th>Resultado</th>
                  <th>Prob.</th>
                  <th>Modelo</th>
                  <th>Sesión</th>
                </tr>
              </thead>
              <tbody>
                {preds.map((p) => (
                  <tr key={p.id}>
                    <td className="text-faint">{p.id}</td>
                    <td className="small">{fmtTs(p.timestamp)}</td>
                    <td className="text-capitalize">{getLabel(p.disease)}</td>
                    <td>
                      <Badge bg={p.prediction === 1 ? 'danger' : 'success'}>
                        {p.prediction === 1 ? 'Riesgo' : 'Bajo riesgo'}
                      </Badge>
                    </td>
                    <td className="num">{pct(p.probability)}</td>
                    <td className="small">{prettyModel(p.model)}</td>
                    <td className="small text-faint">{p.session_id ? p.session_id.slice(0, 8) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        </>
      )}
    </Container>
  );
};

export default Admin;
