// src/components/Logo.jsx
// Logotipo de Predisana: latido + curva de datos dentro de un anillo de
// predicción. Usa currentColor, así que hereda el color del contexto y
// funciona en modo claro y oscuro. Escala a favicon.
const Logo = ({ size = 34, className = '' }) => (
  <svg
    className={className}
    width={size}
    height={size}
    viewBox="0 0 40 40"
    fill="none"
    role="img"
    aria-label="Predisana"
  >
    <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2.5" opacity=".25" />
    <path
      d="M5 22 L13 22 L16 13 L21 28 L24 20 L28 20"
      stroke="currentColor"
      strokeWidth="2.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="28" cy="20" r="3" fill="currentColor" />
  </svg>
);

export default Logo;
