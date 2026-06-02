// src/services/historyService.js
const STORAGE_KEY = "chronic_sim_history_v1";

function safeParse(json, fallback) {
  try {
    const v = JSON.parse(json);
    return v ?? fallback;
  } catch {
    return fallback;
  }
}

export function getHistory() {
  const raw = localStorage.getItem(STORAGE_KEY);
  const arr = safeParse(raw, []);
  return Array.isArray(arr)
    ? arr.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
    : [];
}

export function saveHistoryEntry(entry) {
  const current = getHistory();
  const next = [entry, ...current];
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  return entry;
}

export function deleteHistoryEntry(id) {
  const current = getHistory();
  const next = current.filter((x) => x.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
}

export function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}

export function exportHistoryJSON() {
  const data = getHistory();
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  downloadBlob(blob, `historial_simulaciones_${dateStamp()}.json`);
}

export function exportHistoryCSV() {
  const data = getHistory();
  const csv = toCSV(data);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, `historial_simulaciones_${dateStamp()}.csv`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function dateStamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(
    d.getHours()
  )}-${pad(d.getMinutes())}`;
}

function toCSV(rows) {
  const flat = rows.map((r) => ({
    type: r.type ?? "",
    id: r.id,
    timestamp: new Date(r.timestamp).toISOString(),
    disease: r.disease,
    probability: r.result?.probability ?? "",
    prediction: r.result?.prediction ?? "",
    notes: r.notes ?? "",
    // para comparación, esto puede venir vacío; igual guardamos JSON
    inputs: r.inputs ? JSON.stringify(r.inputs) : "",
    base: r.base ? JSON.stringify(r.base) : "",
    scenarios: r.scenarios ? JSON.stringify(r.scenarios) : "",
    delta: r.delta ? JSON.stringify(r.delta) : "",
    top_features: r.top_features ? JSON.stringify(r.top_features) : "",
    top_features_new: r.top_features_new ? JSON.stringify(r.top_features_new) : "",
  }));

  const headers = Object.keys(flat[0] || {});

  const escape = (v) => {
    const s = String(v ?? "");
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };

  const lines = [
    headers.join(","),
    ...flat.map((row) => headers.map((h) => escape(row[h])).join(",")),
  ];
  return lines.join("\n");
}
