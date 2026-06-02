import axios from 'axios';

// Usamos variable de entorno o fallback a localhost:8000
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const checkHealth = () => api.get('/health');

export const getConfig = (disease) => api.get(`/config/${disease}`);

export const getMetrics = (disease) => api.get(`/metrics/${disease}`);

export const predictRisk = (disease, data) => api.post(`/predict/${disease}`, data);

// ... al final del archivo agrega:
export const getSyntheticCase = (disease) => api.get(`/synthetic/${disease}`);

export default api;