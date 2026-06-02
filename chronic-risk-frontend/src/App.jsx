// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MyNavbar from './components/MyNavbar';
import { Container } from 'react-bootstrap';

import Home from './pages/Home';
import Simulacion from './pages/Simulacion';
import Metricas from './pages/Metricas';
import Aviso from './pages/Aviso';
import Educacion from './pages/Educacion';
import Evaluacion from './pages/Evaluacion';
import Historial from "./pages/Historial";

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
            <Route path="/evaluacion" element={<Evaluacion />} />
            <Route path="/aviso" element={<Aviso />} />
            <Route path="/historial" element={<Historial />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>

        <footer className="cr-footer py-5 mt-auto">
          <Container className="text-center">
            <div className="footer-brand mb-3">
              <img className="footer-logo" src="/icon.jpeg" alt="ChronicRisk AI" />
              <div className="text-start">
                <h5 className="footer-title">CronicApp</h5>
                <div className="footer-sub">Plataforma de predicción de enfermedades crónicas basada en Machine Learning</div>
              </div>
            </div>

            <div className="footer-meta">
              Universidad de Guayaquil | Carrera de Software © {new Date().getFullYear()}
            </div>

            <div className="footer-note">
              Este sistema es puramente educativo y <strong>NO</strong> sustituye un diagnóstico médico.
            </div>
          </Container>
        </footer>
      </div>
    </Router>
  );
}

export default App;
