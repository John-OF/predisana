// Lo que comparten Proyecto y el Laboratorio: las enfermedades (con su icono y su
// color) y el nombre legible de cada algoritmo.
import { Droplet, HeartPulse, Heart, Funnel, Activity } from 'react-bootstrap-icons';

// Desde la v2 todas las enfermedades salen de NHANES 2017-2023 con el mismo esquema
// (cambian los diagnósticos previos: el de la propia enfermedad no está).
export const DISEASE_TABS = [
    { key: 'diabetes', label: 'Diabetes', icon: <Droplet className="me-2" />, variant: 'primary' },
    { key: 'hipertension', label: 'Hipertensión', icon: <HeartPulse className="me-2" />, variant: 'danger' },
    { key: 'cardiovascular', label: 'Cardiovascular', icon: <Heart className="me-2" />, variant: 'info' },
    { key: 'renal', label: 'Renal', icon: <Funnel className="me-2" />, variant: 'success' },
    { key: 'higado', label: 'Hígado graso', icon: <Activity className="me-2" />, variant: 'warning' },
];

export const diseaseLabel = (key) => DISEASE_TABS.find((t) => t.key === key)?.label ?? key;

// Nombres legibles de los algoritmos del leaderboard (claves que emite el backend).
const MODEL_LABELS = {
    logreg: 'Reg. Logística',
    logistic_regression: 'Reg. Logística',
    random_forest: 'Random Forest',
    lightgbm: 'LightGBM',
    xgboost: 'XGBoost',
};
export const prettyModel = (m) => MODEL_LABELS[m] || (m ? String(m) : '—');
