// Avisos de cuánto fiarse del resultado (AUD-16 + coherencia corporal). Los avisos
// tienen la forma EXACTA que devuelve /predict: el fallo que fija este archivo fue
// asumir que todos traían `trained_range`, y el nivel `incoherente` no lo trae.
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

import AvisoSoporte from './AvisoSoporte';

const SIN_DATOS = {
  feature: 'age', value: 90, level: 'sin_datos', trained_range: [29.6, 64.9],
  detail: 'El modelo no vio ningún caso con este valor.',
};
const POCOS_DATOS = {
  feature: 'bmi', value: 48, level: 'pocos_datos', trained_range: [11.1, 69.9],
  common_range: [17.5, 47.2], detail: 'Valor poco frecuente.',
};
const INCOHERENTE = {
  feature: 'waist_circumference', value: 115, level: 'incoherente',
  detail: 'Cintura de 115 cm con un IMC de 19 es una combinación casi imposible fisiológicamente.',
};

describe('AvisoSoporte', () => {
  it('no pinta nada sin avisos', () => {
    const { container } = render(<AvisoSoporte avisos={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('fuera del rango entrenado: cita el rango y avisa de extrapolación', () => {
    render(<AvisoSoporte avisos={[SIN_DATOS]} />);
    expect(screen.getByText(/fuera del rango con datos/i)).toBeInTheDocument();
    expect(screen.getByText(/29\.6–64\.9/)).toBeInTheDocument();
    expect(screen.getByText(/extrapolación/i)).toBeInTheDocument();
  });

  it('un aviso incoherente (sin trained_range) se pinta en vez de romper la página', () => {
    render(<AvisoSoporte avisos={[INCOHERENTE]} />);
    expect(screen.getByText(/no cuadran entre sí/i)).toBeInTheDocument();
    expect(screen.getByText(/casi imposible fisiológicamente/i)).toBeInTheDocument();
    // No es un problema de "pocos datos": no puede salir con ese titular.
    expect(screen.queryByText(/zona con pocos datos/i)).not.toBeInTheDocument();
  });

  it('mezcla de niveles: pinta los dos bloques', () => {
    render(<AvisoSoporte avisos={[POCOS_DATOS, INCOHERENTE]} />);
    expect(screen.getByText(/zona con pocos datos/i)).toBeInTheDocument();
    expect(screen.getByText(/11\.1–69\.9/)).toBeInTheDocument();
    expect(screen.getByText(/no cuadran entre sí/i)).toBeInTheDocument();
  });
});
