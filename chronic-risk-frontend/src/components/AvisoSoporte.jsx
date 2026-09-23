import { Alert } from 'react-bootstrap';
import { getLabel } from '../utils/translations';

// AUD-16 + coherencia corporal: cuánto fiarse del número. El backend lo calcula
// aparte, sin tocar la probabilidad, y cada nivel trae campos distintos:
//   - sin_datos / pocos_datos: una feature fuera del rango visto -> `trained_range`.
//   - incoherente: varios campos que no cuadran ENTRE SÍ (peso/IMC/cintura). No hay
//     un rango que citar, así que solo trae su `detail`.
// Antes el componente asumía `trained_range` en todos: con un aviso `incoherente` la
// desestructuración lanzaba un TypeError y el ErrorBoundary se comía la página.
const AvisoSoporte = ({ avisos }) => {
  if (!avisos?.length) return null;
  const deRango = avisos.filter(a => Array.isArray(a.trained_range));
  const sinRango = avisos.filter(a => !Array.isArray(a.trained_range));
  const extrapola = deRango.some(a => a.level === 'sin_datos');

  return (
    <Alert variant={extrapola || sinRango.length ? 'warning' : 'secondary'} className="py-2 mt-3 text-start">
      {deRango.length > 0 && (
        <small className="d-block">
          <strong>{extrapola ? 'Fuera del rango con datos:' : 'Zona con pocos datos:'}</strong>{' '}
          {deRango.map(a => {
            const [lo, hi] = a.trained_range;
            return `${getLabel(a.feature)} (${a.value}; el modelo aprendió con ${lo}–${hi})`;
          }).join(', ')}.{' '}
          {extrapola
            ? 'Ahí el resultado es una extrapolación: el modelo nunca vio casos así.'
            : 'La estimación es menos fiable de lo habitual en esa zona.'}
        </small>
      )}
      {sinRango.length > 0 && (
        <small className={deRango.length ? 'd-block mt-1' : 'd-block'}>
          <strong>Datos que no cuadran entre sí:</strong>{' '}
          {sinRango.map(a => a.detail || getLabel(a.feature)).join(' ')}
        </small>
      )}
    </Alert>
  );
};

export default AvisoSoporte;
