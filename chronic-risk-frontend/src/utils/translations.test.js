// AUD-19. `getLabel` en si es trivial; lo que merece un test es el CONTRATO con el
// backend: que ninguna feature servida se cuele a la UI sin etiqueta en español y
// acabe pintada como `waist_circumference` en la ficha o en la barra de SHAP. Es un
// fallo silencioso y real: la migracion de hipertension a NHANES (AUD-1) cambio el
// esquema entero, y `diabetes` ya existia en LABELS_ES como nombre de ENFERMEDAD,
// lo que obligo a añadir FIELD_LABEL_OVERRIDES en Simulacion.jsx.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it, expect } from 'vitest';

import { LABELS_ES, getLabel } from './translations';

const MODELOS = join(process.cwd(), '..', 'chronic-risk-backend', 'models');

/** Claves de FIELD_LABEL_OVERRIDES, que resuelve las colisiones por enfermedad. */
function overridesDeSimulacion() {
  const src = readFileSync(join(process.cwd(), 'src', 'pages', 'Simulacion.jsx'), 'utf-8');
  const bloque = src.match(/FIELD_LABEL_OVERRIDES\s*=\s*\{([\s\S]*?)\n\};/);
  if (!bloque) return new Set();
  return new Set([...bloque[1].matchAll(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:/gm)].map(m => m[1]));
}

function tieneEtiqueta(feature, extras) {
  if (feature in LABELS_ES || extras.has(feature)) return true;
  // Las one-hot se pintan como un desplegable con la etiqueta del GRUPO:
  // gender_Male -> "Género", smoking_history_never -> "Historial de Tabaquismo".
  const prefijo = feature.slice(0, feature.lastIndexOf('_'));
  return prefijo in LABELS_ES || extras.has(prefijo);
}

const ficheros = readdirSync(MODELOS).filter(f => f.endsWith('_features.json'));

describe('etiquetas en español', () => {
  it('encuentra los ficheros de features del backend', () => {
    expect(ficheros.length).toBeGreaterThan(0);
  });

  it.each(ficheros)('%s: toda feature servida tiene etiqueta', (fichero) => {
    const extras = overridesDeSimulacion();
    const feats = JSON.parse(readFileSync(join(MODELOS, fichero), 'utf-8'));
    const sinEtiqueta = feats.filter(f => !tieneEtiqueta(f, extras));
    expect(sinEtiqueta, `sin etiqueta en ${fichero}`).toEqual([]);
  });

  it('devuelve la clave tal cual si no la conoce, nunca undefined', () => {
    expect(getLabel('clave_inventada')).toBe('clave_inventada');
    expect(getLabel(undefined)).toBeUndefined();
  });

  it('las tres enfermedades tienen nombre propio', () => {
    for (const d of ['diabetes', 'hipertension', 'cardiovascular']) {
      expect(getLabel(d)).not.toBe(d);
    }
  });
});
