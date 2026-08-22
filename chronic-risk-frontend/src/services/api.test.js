// AUD-19. El session id anonimo (A3) es lo unico con estado del cliente: agrupa las
// simulaciones de una misma persona sin identificarla, y el backend lo sanea a
// [A-Za-z0-9_-] truncado a 64 (AUD-8). Si aqui se generase otro por peticion, la
// analitica del admin contaria una sesion por click.
import { describe, it, expect, vi, beforeEach } from 'vitest';

const CLAVE = 'predisana_session_id';

async function importarApi() {
  vi.resetModules();
  return import('./api');
}

describe('getSessionId', () => {
  beforeEach(() => localStorage.clear());

  it('genera un id y lo persiste', async () => {
    const { getSessionId } = await importarApi();
    const id = getSessionId();
    expect(id).toBeTruthy();
    expect(localStorage.getItem(CLAVE)).toBe(id);
  });

  it('reutiliza el mismo id entre llamadas', async () => {
    const { getSessionId } = await importarApi();
    expect(getSessionId()).toBe(getSessionId());
  });

  it('respeta un id ya guardado (no lo regenera al recargar)', async () => {
    localStorage.setItem(CLAVE, 'id-previo-123');
    const { getSessionId } = await importarApi();
    expect(getSessionId()).toBe('id-previo-123');
  });

  it('sobrevive al saneo del backend: solo [A-Za-z0-9_-] y <=64', async () => {
    const { getSessionId } = await importarApi();
    // Si el id llevara caracteres fuera de ese juego, el backend lo recortaria y dos
    // navegadores distintos podrian colapsar en la misma sesion.
    expect(getSessionId()).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
  });

  it('funciona sin crypto.randomUUID (navegadores viejos / http)', async () => {
    const original = globalThis.crypto;
    // En contexto no seguro crypto.randomUUID no existe: hay un fallback.
    vi.stubGlobal('crypto', {});
    try {
      const { getSessionId } = await importarApi();
      expect(getSessionId()).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
    } finally {
      vi.stubGlobal('crypto', original);
    }
  });
});

describe('predictRisk', () => {
  it('manda el session id en la cabecera X-Session-Id', async () => {
    const api = await importarApi();
    const espia = vi.fn().mockResolvedValue({ data: {} });
    // Se intercepta el post del cliente axios que usa el modulo.
    const cliente = api.default ?? null;
    expect(cliente, 'api.js debe exportar el cliente axios por defecto').not.toBeNull();
    cliente.post = espia;

    await api.predictRisk('diabetes', { age: 40 });

    const [ruta, cuerpo, config] = espia.mock.calls[0];
    expect(ruta).toBe('/predict/diabetes');
    expect(cuerpo).toEqual({ age: 40 });
    expect(config.headers['X-Session-Id']).toBe(api.getSessionId());
  });
});
