// AUD-19. `getLabel` en si es trivial; lo que merece un test es el CONTRATO con el
// backend: que ninguna feature servida se cuele a la UI sin etiqueta en español y
// acabe pintada como `waist_circumference` en la ficha o en la barra de SHAP. Es un
// fallo silencioso y real: la migracion de hipertension a NHANES (AUD-1) cambio el
// esquema entero, y `diabetes` ya existia en LABELS_ES como nombre de ENFERMEDAD,
// lo que obligo a añadir FIELD_LABEL_OVERRIDES en Simulacion.jsx.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it, expect } from 'vitest';

import { LABELS_ES, GRUPOS_ONE_HOT, getFeatureLabel, getLabel } from './translations';

const MODELOS = join(process.cwd(), '..', 'chronic-risk-backend', 'models');
const CURADOS = join(process.cwd(), '..', 'chronic-risk-backend', 'data_curated');

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
  // gender_Male -> "Sexo", smoking_history_never -> "Tabaquismo". Solo ellas: con un
  // prefijo cualquiera, `bmi_autodeclarado` pasaba por la etiqueta de `bmi`.
  const grupo = GRUPOS_ONE_HOT.find(g => feature.startsWith(`${g}_`));
  return grupo !== undefined && (grupo in LABELS_ES || extras.has(grupo));
}

const ficheros = readdirSync(MODELOS).filter(f => f.endsWith('_features.json'));

/** Columnas de las fichas del laboratorio (v2: mas que las de los modelos, como el IMC
 * autodeclarado o la talla): la cabecera del sintetico de cada enfermedad. */
const fichas = readdirSync(CURADOS, { withFileTypes: true })
  .filter(d => d.isDirectory())
  .flatMap(d => readdirSync(join(CURADOS, d.name))
    .filter(f => f.includes('_synthetic'))
    .map(f => [f, readFileSync(join(CURADOS, d.name, f), 'utf-8').split(/\r?\n/, 1)[0].split(',')]));

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

  it('encuentra las fichas del laboratorio', () => {
    expect(fichas.length).toBeGreaterThan(0);
  });

  it.each(fichas)('%s: toda columna de la ficha tiene etiqueta', (fichero, columnas) => {
    const extras = overridesDeSimulacion();
    const sinEtiqueta = columnas.filter(c => c !== 'target' && !tieneEtiqueta(c, extras));
    expect(sinEtiqueta, `sin etiqueta en ${fichero}`).toEqual([]);
  });

  it('devuelve la clave tal cual si no la conoce, nunca undefined', () => {
    expect(getLabel('clave_inventada')).toBe('clave_inventada');
    expect(getLabel(undefined)).toBeUndefined();
  });

  it('una columna one-hot se lee como grupo y opción, no con su clave', () => {
    // Métricas, el laboratorio y el admin pintaban `smoking_history_former` tal cual.
    expect(getFeatureLabel('smoking_history_former')).toBe('Tabaquismo: Exfumador');
    expect(getFeatureLabel('smoking_history_current')).toBe('Tabaquismo: Fumador actual');
    expect(getFeatureLabel('gender_Male')).toBe('Sexo: Masculino');
    expect(getFeatureLabel('bmi_autodeclarado')).toBe('IMC autodeclarado');
    expect(getFeatureLabel('age')).toBe('Edad');
  });

  it('las tres enfermedades tienen nombre propio', () => {
    for (const d of ['diabetes', 'hipertension', 'cardiovascular']) {
      expect(getLabel(d)).not.toBe(d);
    }
  });
});
