// src/components/ErrorBoundary.jsx
// AUD-19: sin esto, un error de render deja la SPA EN BLANCO, sin navbar y sin
// ninguna pista de que ha pasado. React solo captura errores de render con un
// componente de clase (no hay equivalente con hooks), de ahi la clase.
import { Component } from 'react';
import { Button, Container } from 'react-bootstrap';
import { ExclamationTriangle } from 'react-bootstrap-icons';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // La consola es el unico canal: no hay servicio de errores y no se manda nada
    // fuera (las simulaciones son anonimas, ver el aviso de privacidad).
    console.error('[Predisana] error de render:', error, info?.componentStack);
  }

  reintentar = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <Container className="py-5 text-center" style={{ maxWidth: 560 }}>
        <ExclamationTriangle size={40} className="mb-3 text-secondary" />
        <h1 className="h4 mb-3">Algo se rompió en esta página</h1>
        <p className="text-secondary">
          Es un fallo de la interfaz, no de tus datos: no se ha guardado ni enviado nada.
          Puedes reintentar o volver al inicio.
        </p>
        {/* El detalle tecnico solo en desarrollo: al usuario final no le dice nada. */}
        {import.meta.env.DEV && (
          <pre className="text-start small bg-body-secondary p-3 rounded mt-3 overflow-auto">
            {String(error?.stack || error)}
          </pre>
        )}
        <div className="d-flex justify-content-center gap-2 mt-4">
          <Button variant="primary" onClick={this.reintentar}>Reintentar</Button>
          <Button variant="outline-secondary" href="/">Volver al inicio</Button>
        </div>
      </Container>
    );
  }
}

export default ErrorBoundary;
