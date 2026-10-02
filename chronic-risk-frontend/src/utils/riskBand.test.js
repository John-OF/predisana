// Revision 2026-10. Las bandas bajo / moderado / alto ya no son tercios fijos: las
// decide el backend por enfermedad. Lo que se protege aqui es que el front pinte LA
// QUE LLEGA y no vuelva a calcularla por su cuenta: con tercios, una probabilidad de
// diabetes del 31,8% (mas del doble de la media) salia como "Riesgo bajo".
import { describe, it, expect } from 'vitest';

import { riskBand, bandNote } from './riskBand';

describe('riskBand', () => {
  it('pinta la banda que decide el backend, no la de los tercios', () => {
    expect(riskBand({ probability: 0.318, risk_band: 'high' }).label).toBe('Riesgo alto');
    expect(riskBand({ probability: 0.2, risk_band: 'mid' }).label).toBe('Riesgo moderado');
    expect(riskBand({ probability: 0.05, risk_band: 'low' }).label).toBe('Riesgo bajo');
  });

  it('sin risk_band cae a los tercios', () => {
    expect(riskBand({ probability: 0.318 }).key).toBe('low');
    expect(riskBand({ probability: 0.33 }).key).toBe('mid');
    expect(riskBand({ probability: 0.66 }).key).toBe('high');
  });

  it('no revienta con una respuesta vacia o una banda desconocida', () => {
    expect(riskBand(undefined).key).toBe('low');
    expect(riskBand({ probability: 0.9, risk_band: 'altisimo' }).key).toBe('high');
  });
});

describe('bandNote', () => {
  it('explica los cortes relativos a la media de la enfermedad', () => {
    const nota = bandNote({ low_below: 0.1356, high_from: 0.2712, prevalence: 0.1356, relative_to_prevalence: true });
    expect(nota).toContain('13,6 %');
    expect(nota).toContain('al menos al doble (27,1 %)');
  });

  it('explica los tercios cuando la enfermedad es frecuente', () => {
    const nota = bandNote({ low_below: 0.33, high_from: 0.66, prevalence: 0.4973, relative_to_prevalence: false });
    expect(nota).toContain('49,7 %');
    expect(nota).toContain('por debajo del 33 %');
    expect(nota).toContain('al 66 %');
  });

  it('sin prevalencia no dice nada', () => {
    expect(bandNote(undefined)).toBeNull();
    expect(bandNote({ low_below: 0.33, high_from: 0.66, prevalence: null })).toBeNull();
  });
});
