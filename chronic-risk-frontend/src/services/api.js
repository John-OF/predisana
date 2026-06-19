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

export const getAdminStats = (token) => api.get('/admin/stats', adminHeaders(token));

export const getAdminPredictions = (token, { limit = 50, disease } = {}) =>
    api.get('/admin/predictions', { ...adminHeaders(token), params: { limit, disease } });

export default api;