// src/pages/Historial.jsx
import React, { useMemo, useState } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Alert,
  Badge,
  Nav,
  Form,
  Dropdown,
  ButtonGroup,
  Modal,
  Table,
  Toast,
  ToastContainer,
} from "react-bootstrap";

import {
  getHistory,
  deleteHistoryEntry,
  clearHistory,
  exportHistoryJSON,
  exportHistoryCSV,
} from "../services/historyService";

const DISEASE_LABELS = {
  diabetes: "Diabetes",
  hipertension: "Hipertensión",
  obesidad: "Obesidad",
  cardiovascular: "Cardiovascular",
};

const formatPct = (p) => `${(Number(p || 0) * 100).toFixed(1)}%`;

const formatDate = (ts) => {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return "";
  }
};

const prettyKey = (k) =>
  String(k)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const safeNumber = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const fmtVal = (v) => {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    // si tiene decimales, muéstralos
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  return String(v);
};

const buildPairsFromInputs = (inputs = {}) => {
  const entries = Object.entries(inputs || {});
  // orden alfabético para que se vea “limpio”
  entries.sort(([a], [b]) => a.localeCompare(b));
  return entries.map(([k, v]) => ({ key: k, label: prettyKey(k), value: v }));
};

const computeDiffRows = (baseInputs = {}, newInputs = {}) => {
  const keys = new Set([...Object.keys(baseInputs || {}), ...Object.keys(newInputs || {})]);
  const rows = [];
  Array.from(keys)
    .sort((a, b) => a.localeCompare(b))
    .forEach((k) => {
      const b = baseInputs?.[k];
      const n = newInputs?.[k];
      const bn = safeNumber(b);
      const nn = safeNumber(n);

      let delta = null;
      if (bn !== null && nn !== null) delta = nn - bn;

      rows.push({
        key: k,
        label: prettyKey(k),
        base: b,
        next: n,
        delta,
      });
    });

  return rows;
};

async function copyText(text) {
  // clipboard API moderna
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }

  // fallback viejo
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  return ok;
}

function buildSingleSummaryText(item) {
  const diseaseName = DISEASE_LABELS[item.disease] || item.disease || "—";
  const prob = item.result?.probability ?? 0;
  const pred = item.result?.prediction ?? 0;

  const lines = [];
  lines.push(`🧪 Simulación - ${diseaseName}`);
  lines.push(`📅 ${formatDate(item.timestamp)}`);
  lines.push(`📈 Probabilidad: ${formatPct(prob)}`);
  lines.push(`🏷️ Predicción: ${pred === 1 ? "RIESGO ALTO" : "RIESGO BAJO"}`);
  lines.push("");
  lines.push("📝 Inputs:");
  const pairs = buildPairsFromInputs(item.inputs);
  for (const p of pairs) lines.push(`- ${p.label}: ${fmtVal(p.value)}`);
  return lines.join("\n");
}

function buildComparisonSummaryText(item) {
  const diseaseName = DISEASE_LABELS[item.disease] || item.disease || "—";
  const baseProb = item.base?.result?.probability ?? 0;
  const newProb = item.scenarios?.[0]?.result?.probability ?? 0;
  const diffPts = item.delta?.probability_points;

  const direction =
    diffPts === 0
      ? "SIN CAMBIO"
      : diffPts < 0
      ? "MEJORA (↓)"
      : "EMPEORA (↑)";

  const lines = [];
  lines.push(`⚖️ Comparación - ${diseaseName}`);
  lines.push(`📅 ${formatDate(item.timestamp)}`);
  lines.push(`Base: ${formatPct(baseProb)} | Nuevo: ${formatPct(newProb)} | Δ ${diffPts > 0 ? "+" : ""}${Number(diffPts ?? 0).toFixed(1)} pts (${direction})`);
  lines.push("");

  lines.push("🧷 Escenario Base (inputs):");
  const basePairs = buildPairsFromInputs(item.base?.inputs);
  for (const p of basePairs) lines.push(`- ${p.label}: ${fmtVal(p.value)}`);

  lines.push("");
  lines.push("🧩 Escenario Nuevo (inputs):");
  const newPairs = buildPairsFromInputs(item.scenarios?.[0]?.inputs);
  for (const p of newPairs) lines.push(`- ${p.label}: ${fmtVal(p.value)}`);

  return lines.join("\n");
}

export default function Historial() {
  const [refresh, setRefresh] = useState(0);
  const [filterDisease, setFilterDisease] = useState("all");
  const [activeTab, setActiveTab] = useState("all"); // all | single | comparison

  const [selectedItem, setSelectedItem] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const [toast, setToast] = useState({ show: false, msg: "", variant: "success" });

  const allRows = useMemo(() => getHistory(), [refresh]);

  const diseases = useMemo(() => {
    const set = new Set(allRows.map((x) => x.disease).filter(Boolean));
    return ["all", ...Array.from(set)];
  }, [allRows]);

  const rows = useMemo(() => {
    let data = allRows;

    if (filterDisease !== "all") data = data.filter((x) => x.disease === filterDisease);
    if (activeTab !== "all") data = data.filter((x) => (x.type || "single") === activeTab);

    return data;
  }, [allRows, filterDisease, activeTab]);

  const counts = useMemo(() => {
    const c = { all: allRows.length, single: 0, comparison: 0 };
    for (const r of allRows) {
      const t = r.type || "single";
      if (t === "comparison") c.comparison++;
      else c.single++;
    }
    return c;
  }, [allRows]);

  const handleDelete = (id) => {
    deleteHistoryEntry(id);
    setRefresh((x) => x + 1);
  };

  const handleClearAll = () => {
    if (confirm("¿Borrar TODO el historial local de este navegador?")) {
      clearHistory();
      setRefresh((x) => x + 1);
    }
  };

  const openDetails = (item) => {
    setSelectedItem(item);
    setShowModal(true);
  };

  const handleCopy = async (item) => {
    try {
      const t = item.type || "single";
      const text = t === "comparison" ? buildComparisonSummaryText(item) : buildSingleSummaryText(item);
      const ok = await copyText(text);

      setToast({
        show: true,
        msg: ok ? "Copiado al portapapeles ✅" : "No se pudo copiar (bloqueado por el navegador).",
        variant: ok ? "success" : "warning",
      });
    } catch (e) {
      setToast({ show: true, msg: "Error copiando al portapapeles.", variant: "danger" });
    }
  };

  return (
    <Container className="py-4">
      <ToastContainer position="bottom-end" className="p-3">
        <Toast
          bg={toast.variant}
          show={toast.show}
          delay={2200}
          autohide
          onClose={() => setToast((t) => ({ ...t, show: false }))}
        >
          <Toast.Body className={toast.variant === "warning" ? "text-dark" : "text-white"}>
            {toast.msg}
          </Toast.Body>
        </Toast>
      </ToastContainer>

      <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
        <div>
          <h2 className="mb-1 text-primary fw-bold">Historial</h2>
          <div className="text-muted small">
            Guardado localmente en este navegador (no se sube a ningún servidor).
          </div>
        </div>

        <div className="d-flex gap-2 flex-wrap">
          <Button variant="outline-primary" onClick={exportHistoryJSON}>
            ⬇️ JSON
          </Button>
          <Button variant="outline-primary" onClick={exportHistoryCSV}>
            ⬇️ CSV
          </Button>
          <Button variant="outline-danger" onClick={handleClearAll}>
            🗑️ Borrar todo
          </Button>
        </div>
      </div>

      <Card className="shadow-sm border-0 mb-4">
        <Card.Body className="p-3 p-md-4">
          <Row className="g-3 align-items-end">
            <Col md={6}>
              <div className="mb-2 fw-semibold text-muted">Tipo</div>
              <Nav variant="pills" className="gap-2">
                <Nav.Item>
                  <Nav.Link active={activeTab === "all"} onClick={() => setActiveTab("all")}>
                    Todo <Badge bg="light" text="dark" className="ms-1">{counts.all}</Badge>
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link active={activeTab === "single"} onClick={() => setActiveTab("single")}>
                    Simulaciones{" "}
                    <Badge bg="light" text="dark" className="ms-1">{counts.single}</Badge>
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link active={activeTab === "comparison"} onClick={() => setActiveTab("comparison")}>
                    Comparaciones{" "}
                    <Badge bg="light" text="dark" className="ms-1">{counts.comparison}</Badge>
                  </Nav.Link>
                </Nav.Item>
              </Nav>
            </Col>

            <Col md={6}>
              <Form.Group>
                <Form.Label className="fw-semibold text-muted">Filtrar por enfermedad</Form.Label>
                <Form.Select value={filterDisease} onChange={(e) => setFilterDisease(e.target.value)}>
                  {diseases.map((d) => (
                    <option key={d} value={d}>
                      {d === "all" ? "Todas" : (DISEASE_LABELS[d] || d)}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {rows.length === 0 ? (
        <Alert variant="secondary" className="shadow-sm border-0">
          No hay registros guardados para este filtro.
        </Alert>
      ) : (
        <Row className="g-3">
          {rows.map((r) => (
            <Col lg={6} key={r.id}>
              <HistoryCard item={r} onOpen={() => openDetails(r)} onCopy={() => handleCopy(r)} onDelete={() => handleDelete(r.id)} />
            </Col>
          ))}
        </Row>
      )}

      <DetailsModal
        show={showModal}
        onHide={() => setShowModal(false)}
        item={selectedItem}
        onCopy={() => selectedItem && handleCopy(selectedItem)}
      />
    </Container>
  );
}

function HistoryCard({ item, onOpen, onCopy, onDelete }) {
  const diseaseName = DISEASE_LABELS[item.disease] || item.disease || "—";
  const t = item.type || "single";

  const headerBadge =
    t === "comparison" ? (
      <Badge bg="warning" text="dark">⚖️ Comparación</Badge>
    ) : (
      <Badge bg="info" text="dark">🧪 Simulación</Badge>
    );

  return (
    <Card className="shadow-sm border-0 h-100">
      <Card.Body className="p-3 p-md-4">
        <div className="d-flex justify-content-between align-items-start gap-2 mb-2">
          <div>
            <div className="d-flex align-items-center gap-2 flex-wrap">
              <div className="fw-bold text-dark">{diseaseName}</div>
              {headerBadge}
            </div>
            <div className="small text-muted mt-1">{formatDate(item.timestamp)}</div>
          </div>

          <Dropdown as={ButtonGroup} align="end">
            <Dropdown.Toggle variant="outline-secondary" size="sm">⋯</Dropdown.Toggle>
            <Dropdown.Menu>
              <Dropdown.Item onClick={onOpen}>👁️ Ver</Dropdown.Item>
              <Dropdown.Item onClick={onCopy}>📋 Copiar resumen</Dropdown.Item>
              <Dropdown.Divider />
              <Dropdown.Item onClick={onDelete}>🗑️ Eliminar</Dropdown.Item>
            </Dropdown.Menu>
          </Dropdown>
        </div>

        {t === "comparison" ? <ComparisonSummary item={item} /> : <SingleSummary item={item} />}

        <div className="d-flex gap-2 flex-wrap mt-3">
          <Button variant="outline-primary" size="sm" onClick={onOpen}>
            Ver detalles
          </Button>
          <Button variant="outline-secondary" size="sm" onClick={onCopy}>
            Copiar
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
}

function SingleSummary({ item }) {
  const prob = item.result?.probability ?? 0;
  const pred = item.result?.prediction ?? 0;

  const riskBadge =
    pred === 1 ? (
      <Badge bg="danger" className="ms-2">RIESGO ALTO</Badge>
    ) : (
      <Badge bg="success" className="ms-2">RIESGO BAJO</Badge>
    );

  return (
    <>
      <div className="d-flex align-items-baseline gap-2 flex-wrap mt-2">
        <div className="display-6 fw-bold mb-0">{formatPct(prob)}</div>
        {riskBadge}
      </div>

      <div className="text-muted small mt-1">
        {item.top_features?.length ? `Top variables influyentes: ${item.top_features.length}` : "—"}
      </div>
    </>
  );
}

function ComparisonSummary({ item }) {
  const baseProb = item.base?.result?.probability ?? 0;
  const newProb = item.scenarios?.[0]?.result?.probability ?? 0;

  const diffPts = Number(item.delta?.probability_points ?? 0);
  const isImprovement = diffPts < 0;

  const baseLabel = formatPct(baseProb);
  const newLabel = formatPct(newProb);

  const badgeVariant = diffPts === 0 ? "secondary" : isImprovement ? "success" : "warning";

  return (
    <div className="mt-2">
      <div className="d-flex flex-wrap gap-2 align-items-center">
        <Badge bg="secondary">Base: {baseLabel}</Badge>
        <Badge bg={isImprovement ? "success" : "danger"}>Nuevo: {newLabel}</Badge>
        <Badge bg={badgeVariant} text={badgeVariant === "warning" ? "dark" : "white"}>
          Δ {diffPts > 0 ? "+" : ""}{diffPts.toFixed(1)} pts
        </Badge>
      </div>

      <Alert variant={diffPts === 0 ? "secondary" : isImprovement ? "success" : "warning"} className="mt-3 mb-0 py-2">
        <small>
          <strong>Conclusión:</strong>{" "}
          {diffPts === 0
            ? "No hay cambio en el riesgo estimado."
            : isImprovement
              ? "Mejora: el riesgo disminuiría frente al escenario base."
              : "Empeora: el riesgo aumentaría frente al escenario base."}
        </small>
      </Alert>
    </div>
  );
}

function DetailsModal({ show, onHide, item, onCopy }) {
  if (!item) return null;

  const t = item.type || "single";
  const diseaseName = DISEASE_LABELS[item.disease] || item.disease || "—";

  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>
          {t === "comparison" ? "⚖️ Comparación" : "🧪 Simulación"} — {diseaseName}
        </Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <div className="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
          <div className="text-muted small">📅 {formatDate(item.timestamp)}</div>
          <div className="d-flex gap-2">
            <Button variant="outline-secondary" size="sm" onClick={onCopy}>
              📋 Copiar resumen
            </Button>
          </div>
        </div>

        {t === "comparison" ? (
          <ComparisonDetails item={item} />
        ) : (
          <SingleDetails item={item} />
        )}
      </Modal.Body>

      <Modal.Footer>
        <Button variant="outline-secondary" onClick={onHide}>
          Cerrar
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

function SingleDetails({ item }) {
  const prob = item.result?.probability ?? 0;
  const pred = item.result?.prediction ?? 0;

  const pairs = buildPairsFromInputs(item.inputs);

  return (
    <>
      <Alert variant={pred === 1 ? "danger" : "success"} className="py-2">
        <strong>Resultado:</strong> {formatPct(prob)} —{" "}
        {pred === 1 ? "RIESGO ALTO" : "RIESGO BAJO"}
      </Alert>

      <Card className="border-0 shadow-sm">
        <Card.Body>
          <h6 className="text-muted mb-3">📝 Inputs del escenario</h6>
          <Table responsive hover className="mb-0">
            <thead>
              <tr>
                <th style={{ width: "55%" }}>Variable</th>
                <th>Valor</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p) => (
                <tr key={p.key}>
                  <td className="text-muted">{p.label}</td>
                  <td className="fw-semibold">{fmtVal(p.value)}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card.Body>
      </Card>
    </>
  );
}

function ComparisonDetails({ item }) {
  const baseProb = item.base?.result?.probability ?? 0;
  const newProb = item.scenarios?.[0]?.result?.probability ?? 0;

  const diffPts = Number(item.delta?.probability_points ?? 0);
  const isImprovement = diffPts < 0;

  const baseInputs = item.base?.inputs || {};
  const newInputs = item.scenarios?.[0]?.inputs || {};

  const diffRows = computeDiffRows(baseInputs, newInputs);

  return (
    <>
      <Alert variant={diffPts === 0 ? "secondary" : isImprovement ? "success" : "warning"} className="py-2">
        <strong>Probabilidad:</strong>{" "}
        Base {formatPct(baseProb)} → Nuevo {formatPct(newProb)}{" "}
        | <strong>Δ</strong> {diffPts > 0 ? "+" : ""}{diffPts.toFixed(1)} pts
      </Alert>

      <Row className="g-3">
        <Col md={6}>
          <Card className="border-0 shadow-sm h-100">
            <Card.Body>
              <h6 className="text-muted mb-3">🧷 Escenario Base</h6>
              <Table responsive hover className="mb-0">
                <thead>
                  <tr>
                    <th style={{ width: "55%" }}>Variable</th>
                    <th>Valor</th>
                  </tr>
                </thead>
                <tbody>
                  {buildPairsFromInputs(baseInputs).map((p) => (
                    <tr key={p.key}>
                      <td className="text-muted">{p.label}</td>
                      <td className="fw-semibold">{fmtVal(p.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="border-0 shadow-sm h-100">
            <Card.Body>
              <h6 className="text-muted mb-3">🧩 Escenario Nuevo</h6>
              <Table responsive hover className="mb-0">
                <thead>
                  <tr>
                    <th style={{ width: "55%" }}>Variable</th>
                    <th>Valor</th>
                  </tr>
                </thead>
                <tbody>
                  {buildPairsFromInputs(newInputs).map((p) => (
                    <tr key={p.key}>
                      <td className="text-muted">{p.label}</td>
                      <td className="fw-semibold">{fmtVal(p.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card className="border-0 shadow-sm mt-3">
        <Card.Body>
          <h6 className="text-muted mb-3">📊 Diferencias (Nuevo - Base)</h6>
          <Table responsive hover className="mb-0">
            <thead>
              <tr>
                <th style={{ width: "45%" }}>Variable</th>
                <th>Base</th>
                <th>Nuevo</th>
                <th>Diferencia</th>
              </tr>
            </thead>
            <tbody>
              {diffRows.map((r) => {
                const delta = r.delta;
                const deltaText =
                  delta === null ? "—" : (delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2));

                const deltaClass =
                  delta === null ? "" : delta === 0 ? "text-muted" : delta > 0 ? "text-danger" : "text-success";

                return (
                  <tr key={r.key}>
                    <td className="text-muted">{r.label}</td>
                    <td className="fw-semibold">{fmtVal(r.base)}</td>
                    <td className="fw-semibold">{fmtVal(r.next)}</td>
                    <td className={`fw-semibold ${deltaClass}`}>{deltaText}</td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
          <div className="text-muted small mt-2">
            Nota: la diferencia es numérica cuando ambas entradas pueden convertirse a número.
          </div>
        </Card.Body>
      </Card>
    </>
  );
}
