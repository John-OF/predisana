// src/App.jsx
import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container, Spinner } from 'react-bootstrap';
import MyNavbar from './components/MyNavbar';
import Logo from './components/Logo';

import Home from './pages/Home';

// Code-splitting por ruta (AUD-20): recharts pesa ~500 kB y solo lo usan cuatro
// paginas; sweetalert2 solo la simulacion. Cargandolas con lazy(), el bundle
// inicial se queda con el Home y la navegacion, y cada pagina trae su parte al
// entrar. Home NO va lazy a proposito: es la primera pintura.
const Educacion = lazy(() => import('./pages/Educacion'));
const Simulacion = lazy(() => import('./pages/Simulacion'));
const Metricas = lazy(() => import('./pages/Metricas'));
const Proyecto = lazy(() => import('./pages/Proyecto'));
const Aviso = lazy(() => import('./pages/Aviso'));
const Admin = lazy(() => import('./pages/Admin'));

const NotFound = () => (
  <div className="p-5 text-center">
    <h1>404 - Página no encontrada</h1>
  </div>
);

// Placeholder mientras llega el chunk de la pagina.
const CargandoPagina = () => (
  <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '60vh' }}>
    <Spinner animation="border" role="status" variant="secondary">
      <span className="visually-hidden">Cargando…</span>
    </Spinner>
  </div>
);

function App() {
  return (
    <Router>
      <div className="d-flex flex-column min-vh-100">
        <MyNavbar />

        <main className="cr-main flex-grow-1">
          <Suspense fallback={<CargandoPagina />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/educacion" element={<Educacion />} />
              <Route path="/simulacion" element={<Simulacion />} />
              <Route path="/metricas" element={<Metricas />} />
              <Route path="/proyecto" element={<Proyecto />} />
              <Route path="/aviso" element={<Aviso />} />
              {/* Panel dev-only (A3): sin link en navbar, acceso por URL directa + token. */}
              <Route path="/admin" element={<Admin />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </main>

        <footer className="ps-footer mt-auto">
          <Container>
            <div className="foot-in d-flex flex-wrap justify-content-between align-items-center gap-3">
              <div className="foot-brand">
                <Logo className="ps-logo" size={34} />
                <div>
                  <p className="foot-title">Predisana</p>
                  <div className="text-faint">© 2026 John Orellana · IA explicable en salud</div>
                </div>
              </div>
              <div className="mono text-faint" style={{ fontSize: '.8rem' }}>
                React 19 · Vite · LightGBM · SHAP
              </div>
            </div>
            <div className="text-faint mt-3" style={{ fontSize: '.85rem' }}>
              Herramienta educativa. <strong>No</strong> sustituye un diagnóstico médico profesional.
              © {new Date().getFullYear()} Predisana.
            </div>
          </Container>
        </footer>
      </div>
    </Router>
  );
}

export default App;
