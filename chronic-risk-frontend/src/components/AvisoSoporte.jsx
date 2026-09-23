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
  // NHANES registra la edad con tope (80 = "80 o más"): por encima SÍ hubo casos,
  // agrupados en el tope, así que no se puede decir que el modelo "nunca los vio".
  const soloTope = extrapola && deRango.filter(a => a.level === 'sin_datos').every(a => a.topcoded != null);

  return (
    <Alert variant={extrapola || sinRango.length ? 'warning' : 'secondary'} className="py-2 mt-3 text-start">
      {deRango.length > 0 && (
        <small className="d-block">
          <strong>{extrapola ? 'Fuera del rango con datos:' : 'Zona con pocos datos:'}</strong>{' '}
          {deRango.map(a => {
            if (a.topcoded != null) {
              return `${getLabel(a.feature)} (${a.value}; en los datos, todo el que pasa de ${a.topcoded} figura como ${a.topcoded})`;
            }
            const [lo, hi] = a.trained_range;
            return `${getLabel(a.feature)} (${a.value}; el modelo aprendió con ${lo}–${hi})`;
          }).join(', ')}.{' '}
          {soloTope
            ? 'El modelo sí vio casos así, pero agrupados en ese tope: por encima, el resultado prolonga la tendencia que aprendió.'
            : extrapola
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
