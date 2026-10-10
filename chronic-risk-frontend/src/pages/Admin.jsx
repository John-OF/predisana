import { Fragment, useState, useEffect, useCallback } from 'react';
import { Container, Row, Col, Form, Button, Table, Spinner, Alert, Badge } from 'react-bootstrap';
import { ShieldLock, BoxArrowRight, ArrowClockwise, Download, FunnelFill } from 'react-bootstrap-icons';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell,
} from 'recharts';
import {
  verifyAdmin, getAdminStats, getAdminPredictions, downloadAdminCsv,
} from '../services/api';
import { getLabel, getFeatureLabel } from '../utils/translations';
import { riskBand } from '../utils/riskBand';
import { prettyModel } from '../utils/catalogo';

// El token vive en sessionStorage: se borra al cerrar la pestaña (más seguro que
// localStorage para una credencial de admin). NO es auth de usuario.
const TOKEN_KEY = 'predisana_admin_token';

const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);
// La API manda la hora en UTC con su Z: el navegador la pasa a la hora local.
const fmtTs = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

// Desfase horario del navegador frente a UTC, en horas (Ecuador: -5). Las horas del
// servidor van en UTC; si el desfase no es entero (India, +5:30) se dejan en UTC.
const DESFASE = -new Date().getTimezoneOffset() / 60;
const HORA_LOCAL = Number.isInteger(DESFASE);

// Rellena con ceros los días sin simulaciones: sin ellos, el área unía dos días
// separados por una semana como si hubiera una tendencia entre ellos.
const conDiasVacios = (timeline) => {
  if (timeline.length < 2) return timeline;
  const cuenta = Object.fromEntries(timeline.map((t) => [t.day, t.count]));
  const out = [];
  const fin = new Date(`${timeline[timeline.length - 1].day}T00:00:00Z`);
  for (let d = new Date(`${timeline[0].day}T00:00:00Z`); d <= fin; d.setUTCDate(d.getUTCDate() + 1)) {
    const dia = d.toISOString().slice(0, 10);
    out.push({ day: dia, count: cuenta[dia] || 0 });
  }
  return out;
};

// Etiqueta de una variable de la ficha guardada (grupos one-hot incluidos).
const SI_NO = ['hypertension', 'diabetes', 'high_cholesterol', 'heart_disease'];
const valorFicha = (k, v) => (k.startsWith('gender_') || k.startsWith('smoking_history_') || SI_NO.includes(k)
  ? (Number(v) === 1 ? 'sí' : 'no') : String(v));

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
  const [abierta, setAbierta] = useState(null);   // fila del historial desplegada

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
      } else if (err?.response?.status === 429) {
        setDataError('Demasiadas peticiones al panel. Espera un momento y vuelve a intentar.');
      } else if (err?.response?.status === 403) {
        setDataError('Este origen no está autorizado para el panel admin (ADMIN_CORS_ORIGINS en el servidor).');
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
      } else if (err?.response?.status === 429) {
        // AUD-4: el servidor corta los intentos repetidos de token (anti fuerza bruta).
        setAuthError('Demasiados intentos. Espera un minuto antes de volver a probar.');
      } else if (err?.response?.status === 403) {
        setAuthError('Este origen no está autorizado para el panel admin (ADMIN_CORS_ORIGINS en el servidor).');
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
  const timeline = conDiasVacios(stats?.timeline || []);
  const maxCount = Math.max(...byDisease.map(d => d.count || 0), 1);
  const consultadas = byDisease.filter(d => d.count > 0).length;

  // Uso por hora del día (0-23), pasado a la hora local si el desfase es entero.
  const hourlyUtc = stats?.hourly || [];
  const hourlyData = hourlyUtc.map((_, h) => ({
    hour: String(h).padStart(2, '0'),
    count: HORA_LOCAL ? hourlyUtc[(((h - DESFASE) % 24) + 24) % 24] : hourlyUtc[h],
  }));

  // Histograma de probabilidad para la enfermedad elegida (10 bins 0..1).
  const nBins = stats?.prob_bins || 10;
  const diseasesWithData = byDisease.filter(d => d.count > 0).map(d => d.disease);
  const activeHistDisease = histDisease || diseasesWithData[0] || (byDisease[0]?.disease);
  // Cada franja, del color de la banda que ve el usuario en esa enfermedad (por su
  // punto medio): antes, rojo desde el 50% para todas.
  const cortes = byDisease.find(d => d.disease === activeHistDisease)?.risk_bands;
  const histBins = (stats?.prob_histogram?.[activeHistDisease] || []).map((c, i) => {
    const medio = (i + 0.5) / nBins;
    const banda = !cortes ? 'mid' : medio < cortes.low_below ? 'low' : medio >= cortes.high_from ? 'high' : 'mid';
    return {
      band: `${Math.round((i / nBins) * 100)}–${Math.round(((i + 1) / nBins) * 100)}%`,
      count: c,
      color: { low: '#4fae8c', mid: '#d8a64a', high: '#c8736a' }[banda],
    };
  });

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
                <div className="k-val">{consultadas}</div>
                <div className="k-sub">consultadas, de {byDisease.length}</div>
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
              <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Simulaciones por día <span className="text-faint small">(días UTC)</span></h3>
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
              <p className="text-soft small mb-2">
                Cuántas simulaciones caen en cada franja de probabilidad (la calibrada, la que ve el usuario) para{' '}
                {getLabel(activeHistDisease)}, con el color de su banda: verde bajo, ocre moderado, rojo alto.
              </p>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={histBins} margin={{ top: 6, right: 12, left: -12, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="band" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={50} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Simulaciones" radius={[4, 4, 0, 0]}>
                    {histBins.map((b, i) => (
                      <Cell key={i} fill={b.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Col>
            <Col lg={6}>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '8px' }}>Simulaciones por hora del día</h3>
              <p className="text-soft small mb-2">
                Cuándo se usa el simulador ({HORA_LOCAL ? 'hora local de este navegador' : 'hora UTC'}, 0–23).
              </p>
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
                        <td>{getFeatureLabel(f.feature)}</td>
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
                  <th title="Banda que vio el usuario">Bajo · Moderado · Alto</th>
                  <th>Prob. media</th>
                  <th style={{ width: '24%' }}>Volumen</th>
                </tr>
              </thead>
              <tbody>
                {byDisease.map((d) => (
                  <tr key={d.disease}>
                    <td className="text-capitalize">{getLabel(d.disease)}</td>
                    <td className="small">
                      {d.models
                        ? Object.entries(d.models).map(([modo, m]) => (
                            <div key={modo}>{getLabel(modo)}: {prettyModel(m)}</div>
                          ))
                        : prettyModel(d.model)}
                    </td>
                    <td className="num">
                      {d.count}
                      {d.modes && d.count > 0 && (
                        <div className="small text-faint">{d.modes.simplificado} simpl. · {d.modes.completo} compl.</div>
                      )}
                    </td>
                    <td className="num small">
                      {d.bands && d.count > 0
                        ? ['low', 'mid', 'high'].map(b => pct(d.bands[b] / d.count, 0)).join(' · ')
                        : '—'}
                    </td>
                    <td className="num text-faint">{pct(d.avg_probability)}</td>
                    <td><div className="ps-mini-bar" style={{ width: `${Math.round((d.count / maxCount) * 100)}%` }} /></td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>

          {/* Predicciones recientes */}
          <h3 style={{ fontSize: '1.3rem', marginBottom: '14px' }}>Simulaciones recientes</h3>
          <p className="text-soft small mb-3">
            Últimas {preds.length} (máx. 50). Anónimas: el <code>session_id</code> es un UUID aleatorio sin PII.
            Pulsa una fila para ver los datos que se enviaron y la nota clínica.
          </p>
          <div className="table-responsive">
            <Table className="align-middle" size="sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Fecha</th>
                  <th>Enfermedad</th>
                  <th>Banda</th>
                  <th>Prob.</th>
                  <th>Modelo</th>
                  <th>Sesión</th>
                </tr>
              </thead>
              <tbody>
                {preds.map((p) => {
                  const banda = riskBand(p);
                  return (
                    <Fragment key={p.id}>
                      <tr onClick={() => setAbierta(abierta === p.id ? null : p.id)} style={{ cursor: 'pointer' }}>
                        <td className="text-faint">{p.id}</td>
                        <td className="small">{fmtTs(p.timestamp)}</td>
                        <td className="text-capitalize">{getLabel(p.disease)}</td>
                        <td>
                          <span className="ps-risk-pill" style={{ background: banda.bg, color: banda.color, fontSize: '.75rem' }}>{banda.label}</span>
                        </td>
                        <td className="num">{pct(p.probability)}</td>
                        <td className="small">{prettyModel(p.model)}{p.mode ? ` · ${getLabel(p.mode)}` : ''}</td>
                        <td className="small text-faint">{p.session_id ? p.session_id.slice(0, 8) : '—'}</td>
                      </tr>
                      {abierta === p.id && (
                        <tr>
                          <td colSpan={7} className="small">
                            <div className="d-flex flex-wrap gap-2 mb-2">
                              {Object.entries(p.input_data || {}).map(([k, v]) => (
                                <Badge key={k} bg="light" text="dark" className="fw-normal">
                                  {getFeatureLabel(k)}: {valorFicha(k, v)}
                                </Badge>
                              ))}
                            </div>
                            <div className="text-soft">{p.clinical_note || 'Sin nota clínica.'}</div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </Table>
          </div>
        </>
      )}
    </Container>
  );
};

export default Admin;
