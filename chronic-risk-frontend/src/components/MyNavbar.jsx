import { Navbar, Nav, Container } from 'react-bootstrap';
import { NavLink } from 'react-router-dom';
import { SunFill, MoonStarsFill } from 'react-bootstrap-icons';
import Logo from './Logo';
import { useTheme } from '../context/ThemeContext';

const MyNavbar = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <Navbar expand="lg" className="ps-nav">
      <Container>
        <Navbar.Brand as={NavLink} to="/" className="ps-brand">
          <Logo className="ps-logo" size={32} />
          Predisana
        </Navbar.Brand>

        <div className="d-flex align-items-center gap-2 order-lg-last">
          <button
            type="button"
            className="theme-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Activar modo claro' : 'Activar modo oscuro'}
            title={theme === 'dark' ? 'Modo claro' : 'Modo oscuro'}
          >
            {theme === 'dark' ? <SunFill size={18} /> : <MoonStarsFill size={18} />}
          </button>
          <Navbar.Toggle aria-controls="ps-navbar-nav" />
        </div>

        <Navbar.Collapse id="ps-navbar-nav">
          <Nav className="ms-auto align-items-lg-center">
            <Nav.Link as={NavLink} to="/" end className="navlink">
              Inicio
            </Nav.Link>
            <Nav.Link as={NavLink} to="/educacion" className="navlink">
              Educación
            </Nav.Link>
            <Nav.Link as={NavLink} to="/simulacion" className="navlink">
              Simulador
            </Nav.Link>
            <Nav.Link as={NavLink} to="/metricas" className="navlink">
              Métricas
            </Nav.Link>
            <Nav.Link as={NavLink} to="/proyecto" className="navlink">
              Acerca de
            </Nav.Link>
            <Nav.Link as={NavLink} to="/aviso" className="navlink">
              Aviso Legal
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default MyNavbar;
