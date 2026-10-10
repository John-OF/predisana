// Banda de riesgo serena (no alarmista): bajo / moderado / alto.
// Los cortes no son fijos: los decide el backend por enfermedad (`risk_band` y
// `risk_bands` en /predict), porque un 30% no significa lo mismo en diabetes, donde
// la media de los datos es 18,9%, que en hipertension, donde es 44%. Con tercios
// fijos, en la v1 una glucosa de 250 salia como "Riesgo bajo". Aqui solo se pinta.
const BANDAS = {
  low: { key: 'low', label: 'Riesgo bajo', color: 'var(--risk-low)', bg: 'var(--risk-low-bg)' },
  mid: { key: 'mid', label: 'Riesgo moderado', color: 'var(--risk-mid)', bg: 'var(--risk-mid-bg)' },
  high: { key: 'high', label: 'Riesgo alto', color: 'var(--risk-high)', bg: 'var(--risk-high-bg)' },
};

/** Banda de una respuesta de /predict. Sin `risk_band` (backend antiguo) se cae a
 *  los tercios de siempre. */
export const riskBand = (result) => {
  if (BANDAS[result?.risk_band]) return BANDAS[result.risk_band];
  const pct = (result?.probability || 0) * 100;
  if (pct < 33) return BANDAS.low;
  return pct < 66 ? BANDAS.mid : BANDAS.high;
};

/** Porcentaje de riesgo para mostrar (0-100 -> texto). Redondeado a entero, un 0,2%
 *  salia como "0%": un riesgo nulo que ningun modelo puede prometer. */
export const pctRiesgo = (pct) => {
  if (!Number.isFinite(pct)) return '—';
  if (pct < 1) return '<1%';
  if (pct > 99) return '>99%';
  return `${Math.round(pct)}%`;
};

const comoPorcentaje = (v) => `${(v * 100).toFixed(1).replace(/\.0$/, '').replace('.', ',')} %`;

/** Frase que cuenta de donde salen los cortes (`risk_bands` de /predict), o null. */
export const bandNote = (bands) => {
  if (bands?.prevalence == null) return null;
  const media = `En los datos de entrenamiento, el ${comoPorcentaje(bands.prevalence)} de las personas figura con la enfermedad.`;
  return bands.relative_to_prevalence
    ? `${media} «Bajo» es quedar por debajo de esa media; «alto», llegar al menos al doble (${comoPorcentaje(bands.high_from)}).`
    : `${media} «Bajo» es quedar por debajo del ${comoPorcentaje(bands.low_below)}; «alto», llegar al ${comoPorcentaje(bands.high_from)}.`;
};
