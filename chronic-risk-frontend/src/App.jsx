// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container } from 'react-bootstrap';
import MyNavbar from './components/MyNavbar';
import Logo from './components/Logo';

import Home from './pages/Home';
import Simulacion from './pages/Simulacion';
import Metricas from './pages/Metricas';
import Aviso from './pages/Aviso';
import Educacion from './pages/Educacion';
import Proyecto from './pages/Proyecto';

const NotFound = () => (
  <div className="p-5 text-center">
    <h1>404 - Página no encontrada</h1>
  </div>
);

function App() {
  return (
    <Router>
      <div className="d-flex flex-column min-vh-100">
        <MyNavbar />

        <main className="cr-main flex-grow-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/educacion" element={<Educacion />} />
            <Route path="/simulacion" element={<Simulacion />} />
            <Route path="/metricas" element={<Metricas />} />
            <Route path="/proyecto" element={<Proyecto />} />
            <Route path="/aviso" element={<Aviso />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>

        <footer className="ps-footer mt-auto">
          <Container>
            <div className="foot-in d-flex flex-wrap justify-content-between align-items-center gap-3">
              <div className="foot-brand">
                <Logo className="ps-logo" size={34} />
                <div>
                  <p className="foot-title">Predisana</p>
                  <div className="text-faint">Proyecto desarrollado para cliente privado · IA explicable en salud</div>
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
