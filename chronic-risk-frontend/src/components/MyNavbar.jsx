import { useEffect, useState } from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { NavLink } from 'react-router-dom';

const MyNavbar = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <Navbar
      expand="lg"
      variant="dark"
      className={`custom-navbar fixed-top ${scrolled ? 'nav-scrolled' : ''}`}
    >
      <Container>
        <Navbar.Brand as={NavLink} to="/" className="d-flex align-items-center fw-bold">
          <img
            src="/icon.jpeg"
            width="40"
            height="40"
            className="d-inline-block align-top me-2 rounded-circle border border-info"
            alt="Logo"
          />
          CronicApp
        </Navbar.Brand>

        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link as={NavLink} to="/" end className="px-3 navlink">
              Inicio
            </Nav.Link>
            <Nav.Link as={NavLink} to="/educacion" className="px-3 navlink">
              Educación
            </Nav.Link>
            <Nav.Link as={NavLink} to="/simulacion" className="px-3 navlink">
              Simulador IA
            </Nav.Link>
            <Nav.Link as={NavLink} to="/evaluacion" className="px-3 navlink">
              Evaluación
            </Nav.Link>
            <Nav.Link as={NavLink} to="/metricas" className="px-3 navlink">
              Métricas
            </Nav.Link>
            <Nav.Link as={NavLink} to="/aviso" className="px-3 navlink">
              Aviso Legal
            </Nav.Link>
            <Nav.Link as={NavLink} to="/historial" className="px-3 navlink">
              Historial
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default MyNavbar;
