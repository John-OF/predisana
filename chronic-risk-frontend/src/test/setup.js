// Arranque de la suite de frontend (AUD-19).
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  cleanup();                 // desmonta lo renderizado: los tests no se contaminan
  localStorage.clear();      // getSessionId() persiste un UUID; que no cruce tests
  vi.restoreAllMocks();
});
