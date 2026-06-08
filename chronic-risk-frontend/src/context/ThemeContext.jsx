// src/context/ThemeContext.jsx
// Tema claro/oscuro de "Pulso Sereno". Aplica data-theme al <html> y lo
// persiste en localStorage. Compartido por toda la app vía contexto.
import { createContext, useContext, useEffect, useState } from 'react';

const STORAGE_KEY = 'predisana_theme_v1';
const ThemeContext = createContext(null);

function getInitialTheme() {
  if (typeof window === 'undefined') return 'light';
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  return 'light'; // claro por defecto (proyecto educativo, sereno)
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    // Sincroniza el modo nativo de Bootstrap 5.3 para que los componentes
    // de react-bootstrap (forms, dropdowns, etc.) tomen el contraste correcto.
    root.setAttribute('data-bs-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'));

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme debe usarse dentro de <ThemeProvider>');
  return ctx;
}
