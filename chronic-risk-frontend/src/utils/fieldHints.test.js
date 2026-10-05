// Revision 2026-10 (v2). El paso de cada campo del simulador tiene que admitir la
// precision del dato real. «Caso virtual» carga en el formulario una ficha del
// sintetico, con los decimales del real, y un input con un paso mas grueso que el dato
// queda invalido: el navegador bloquea el envio sin que el formulario diga nada. Paso
// con la presion de NHANES (media de tres lecturas: 101,4), la talla (de pulgadas:
// 172,7), el eGFR y la albumina en orina (dos decimales).
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it, expect } from 'vitest';

import { FIELD_HINTS } from './fieldHints';

const CURADOS = join(process.cwd(), '..', 'chronic-risk-backend', 'data_curated');

const decimales = (x) => {
  const s = String(Number(x));
  return s.includes('.') ? s.split('.')[1].length : 0;
};

const sinteticos = readdirSync(CURADOS, { withFileTypes: true })
  .filter(d => d.isDirectory())
  .flatMap(d => readdirSync(join(CURADOS, d.name))
    .filter(f => f.includes('_synthetic'))
    .map(f => [f, join(CURADOS, d.name, f)]));

describe('paso de los campos del simulador', () => {
  it('encuentra las fichas del laboratorio', () => {
    expect(sinteticos.length).toBeGreaterThan(0);
  });

  it.each(sinteticos)('%s: cada valor cabe en el paso de su campo', (fichero, ruta) => {
    const [cabecera, ...filas] = readFileSync(ruta, 'utf-8').trim().split(/\r?\n/);
    const demasiados = {};
    cabecera.split(',').forEach((col, j) => {
      const hint = FIELD_HINTS[col];
      if (!hint) return;
      const peor = Math.max(...filas.map(f => decimales(f.split(',')[j])));
      if (peor > decimales(hint.step)) demasiados[col] = `${peor} decimales con paso ${hint.step}`;
    });
    expect(demasiados, `en ${fichero}`).toEqual({});
  });

  it('los sí/no y las cuentas enteras siguen yendo de uno en uno', () => {
    for (const f of ['age', 'hypertension', 'diabetes', 'heart_disease', 'high_cholesterol']) {
      expect(FIELD_HINTS[f].step, f).toBe(1);
    }
  });
});
