import axios from 'axios';

// Usamos variable de entorno o fallback a localhost:8000
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// UUID anónimo por navegador (A3): agrupa simulaciones de una misma persona sin
// identificarla. No es PII; se genera la primera vez y se persiste en localStorage.
const SESSION_KEY = 'predisana_session_id';
export function getSessionId() {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
        id = (typeof crypto !== 'undefined' && crypto.randomUUID)
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        localStorage.setItem(SESSION_KEY, id);
    }
    return id;
}

export const checkHealth = () => api.get('/health');

export const getConfig = (disease) => api.get(`/config/${disease}`);

export const getMetrics = (disease) => api.get(`/metrics/${disease}`);

export const predictRisk = (disease, data) =>
    api.post(`/predict/${disease}`, data, { headers: { 'X-Session-Id': getSessionId() } });

// Análisis contrafactual: barre una feature sobre un rango y devuelve la curva de
// riesgo (probabilidad calibrada). NO se registra en la BD.
export const getWhatIf = (disease, { base, feature, min, max, steps = 25 }) =>
    api.post(`/whatif/${disease}`, { base, feature, min, max, steps });

// ... al final del archivo agrega:
export const getSyntheticCase = (disease) => api.get(`/synthetic/${disease}`);

// Ficha de paciente del origen pedido: source = 'real' | 'synthetic'.
export const getSampleCase = (disease, source = 'synthetic') =>
    api.get(`/sample/${disease}`, { params: { source } });

// Histograma comparado real vs sintético de una variable numérica.
export const getDistribution = (disease, feature) =>
    api.get(`/distribution/${disease}`, { params: { feature } });

// Calidad del sintético: score SDMetrics + matrices de correlación.
export const getSyntheticQuality = (disease) =>
    api.get(`/synthetic_quality/${disease}`);

// ===== Panel admin (dev-only — A3) =====
// El token se manda en header X-Admin-Token. No es auth de usuario.
const adminHeaders = (token) => ({ headers: { 'X-Admin-Token': token } });

export const verifyAdmin = (token) => api.get('/admin/verify', adminHeaders(token));

export const getAdminStats = (token, { from, to } = {}) =>
    api.get('/admin/stats', { ...adminHeaders(token), params: { from, to } });

export const getAdminPredictions = (token, { limit = 50, disease, from, to } = {}) =>
    api.get('/admin/predictions', { ...adminHeaders(token), params: { limit, disease, from, to } });

// URL de descarga del CSV (el token va como query para poder usarlo en un <a download>).
export const adminExportCsvUrl = (token, { disease, from, to } = {}) => {
    const params = new URLSearchParams();
    if (disease) params.set('disease', disease);
    if (from) params.set('from', from);
    if (to) params.set('to', to);
    // El backend valida el token por header; para <a> lo pasamos por fetch abajo.
    return `${API_URL}/admin/export.csv?${params.toString()}`;
};

// Descarga el CSV vía fetch (para poder mandar el header de token) y dispara el save.
export const downloadAdminCsv = async (token, opts = {}) => {
    const res = await fetch(adminExportCsvUrl(token, opts), {
        headers: { 'X-Admin-Token': token },
    });
    if (!res.ok) throw new Error(`export falló: ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predisana_simulaciones.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
};

export default api;