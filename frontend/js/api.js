// Thin API client: in-memory cache, latency tracking, error surfacing.
const cache = new Map();
const latencies = [];

function reportLatency(ms, serverMs) {
  latencies.push(ms);
  if (latencies.length > 20) latencies.shift();
  const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const el = document.getElementById("api-latency");
  if (el) el.textContent = `last ${ms.toFixed(0)} ms · server ${serverMs ?? "–"} ms · avg ${avg.toFixed(0)} ms`;
}

export async function api(path, { method = "GET", body, fresh = false } = {}) {
  const key = method + path + (body ? JSON.stringify(body) : "");
  if (!fresh && method === "GET" && cache.has(key)) return cache.get(key);
  const t0 = performance.now();
  const promise = fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).then(async (r) => {
    reportLatency(performance.now() - t0, r.headers.get("X-Process-Time-ms"));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(detail.detail || `${r.status} ${r.statusText}`);
    }
    return r.json();
  });
  if (method === "GET") {
    cache.set(key, promise);
    promise.catch(() => cache.delete(key));
  }
  return promise;
}

export function liveSocket(params) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const qs = new URLSearchParams(params).toString();
  return new WebSocket(`${proto}://${location.host}/ws/live?${qs}`);
}
