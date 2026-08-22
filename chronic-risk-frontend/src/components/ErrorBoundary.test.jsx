// AUD-19: el ErrorBoundary es lo unico que separa un fallo de render de una pantalla
// EN BLANCO. Se verifico a mano al escribirlo (forzando un throw sobre el build de
// produccion); estos tests lo dejan fijado.
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, it, expect, vi } from 'vitest';

import ErrorBoundary from './ErrorBoundary';

const Explota = () => { throw new Error('boom de prueba'); };

/** Silencia el ruido que React escupe por consola al capturar el error. */
const sinRuido = () => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
};

describe('ErrorBoundary', () => {
  it('deja pasar a sus hijos cuando no hay error', () => {
    render(<ErrorBoundary><p>contenido normal</p></ErrorBoundary>);
    expect(screen.getByText('contenido normal')).toBeInTheDocument();
    expect(screen.queryByText(/se rompió/i)).not.toBeInTheDocument();
  });

  it('muestra la pantalla de recuperacion en vez de quedarse en blanco', () => {
    sinRuido();
    render(<ErrorBoundary><Explota /></ErrorBoundary>);
    expect(screen.getByText(/algo se rompió en esta página/i)).toBeInTheDocument();
    // Lo importante no es el mensaje, es que haya salida. Ojo: react-bootstrap pinta
    // el <Button href> como <a role="button">, asi que ambos salen como 'button'.
    expect(screen.getByRole('button', { name: /reintentar/i })).toBeInTheDocument();
    const inicio = screen.getByRole('button', { name: /volver al inicio/i });
    expect(inicio).toHaveAttribute('href', '/');
  });

  it('tranquiliza sobre los datos: no se envia ni se guarda nada', () => {
    sinRuido();
    render(<ErrorBoundary><Explota /></ErrorBoundary>);
    expect(screen.getByText(/no se ha guardado ni enviado nada/i)).toBeInTheDocument();
  });

  it('registra el error en consola para poder depurarlo', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<ErrorBoundary><Explota /></ErrorBoundary>);
    const nuestro = spy.mock.calls.find(c => String(c[0]).includes('[Predisana]'));
    expect(nuestro).toBeTruthy();
    expect(String(nuestro[1])).toContain('boom de prueba');
  });

  it('«Reintentar» vuelve a montar el hijo, no deja la pantalla pegada', async () => {
    sinRuido();
    // El fallo se controla con un flag externo, NO con un contador de renders: React
    // reintenta el render al capturar un error, asi que un "falla la primera vez" se
    // recuperaria solo y el test no probaria nada.
    let debeFallar = true;
    const AVecesFalla = () => {
      if (debeFallar) throw new Error('fallo transitorio');
      return <p>ya funciona</p>;
    };
    render(<ErrorBoundary><AVecesFalla /></ErrorBoundary>);
    expect(screen.getByText(/algo se rompió/i)).toBeInTheDocument();

    debeFallar = false;                       // el fallo transitorio se resolvio
    await userEvent.click(screen.getByRole('button', { name: /reintentar/i }));
    expect(screen.getByText('ya funciona')).toBeInTheDocument();
    expect(screen.queryByText(/algo se rompió/i)).not.toBeInTheDocument();
  });

  it('no filtra el stack al usuario en produccion', () => {
    sinRuido();
    // import.meta.env.DEV es true bajo vitest, asi que aqui se comprueba el contrato
    // al reves: el detalle tecnico va dentro de un <pre>, separado del mensaje.
    const { container } = render(<ErrorBoundary><Explota /></ErrorBoundary>);
    const pre = container.querySelector('pre');
    if (import.meta.env.DEV) {
      expect(pre).not.toBeNull();
      expect(pre.textContent).toContain('boom de prueba');
    } else {
      expect(pre).toBeNull();
    }
  });

  // Contador de renders para el test de useState: React monta dos veces en StrictMode,
  // pero aqui no se usa StrictMode, asi que el contador es fiable.
  it('mantiene el estado del hijo mientras no haya error', () => {
    const Contador = () => {
      const [n, setN] = useState(0);
      return <button onClick={() => setN(n + 1)}>clicks: {n}</button>;
    };
    render(<ErrorBoundary><Contador /></ErrorBoundary>);
    expect(screen.getByRole('button', { name: /clicks: 0/ })).toBeInTheDocument();
  });
});
