import { api } from "../api.js";

const ENDPOINTS = [
  ["GET", "/api/health", "Liveness + pipeline timings"],
  ["GET", "/api/overview", "KPIs, top scorers, over-performers"],
  ["GET", "/api/league/table", "Standings with xG, xPts, Elo, form"],
  ["GET", "/api/league/elo", "Elo history per team"],
  ["GET", "/api/league/ratings", "Dixon-Coles ratings + ground-truth recovery"],
  
  ["GET", "/api/matches?matchday=1", "Fixtures (played & upcoming with probabilities)"],
  
  ["GET", "/api/models/xg", "xG report: coefficients, calibration, ROC"],
  ["POST", "/api/models/xg/predict", "Score arbitrary shots"],
  ["GET", "/api/models/xg/surface", "xG evaluated over a pitch grid"],
  ["GET", "/api/models/xt", "xT grid + value-iteration snapshots"],
  ["GET", "/api/teams", "All teams (ids for the endpoints below)"],
  ["GET", "/api/simulate/season?n=2000&as_of=20", "Monte Carlo from any matchday"],
  ["GET", "/api/models/backtest", "Walk-forward RPS backtest"],
  ["GET", "/api/players/map", "PCA coordinates + k-means roles"],
  
  ["GET", "/api/simulate/title-race", "Title odds after every matchday"],
  ["GET", "/api/models/halfway", "Predict the 2nd half of the season"],
  ["WS", "/ws/live?match_id=…&speed=5", "Minute-by-minute match replay (JSON)"],
];

export async function render(el) {
  const o = await api("/api/overview");
  const real = o.source && o.source.source === "statsbomb";
  const t = o.timings;
  el.innerHTML = `
    <div class="card">
      <div class="card-head"><div><h3 class="card-title">Data &amp; model pipeline</h3><div class="card-sub">Runs once at start-up in ${t.total}s; results are cached in memory and served from the API</div></div></div>
      <div class="pipeline">
        ${[
          real ? ["statsbomb.py", "1 · Ingestion", `Streams ${o.source.competition} from StatsBomb open data (380 match files + lineups, 8 parallel workers), keeps only modelled event types and maps them to one schema.`]
               : ["engine.py", "1 · Match engine", "Markov-chain possession simulator generates StatsBomb-style events (passes, carries, shots, tackles) with hidden ground-truth probabilities."],
          ["league.py → SQLite", "2 · Event store", `${o.kpis.events.toLocaleString()} events in an indexed SQLite DB, loaded with pandas in ${t.load}s.`],
          ["xg.py · xt.py", "3 · Action value models", `xG logistic regression (IRLS) in ${t.xg}s; xT value iteration in ${t.xt}s. Every action gets a value.`],
          ["ratings.py", "4 · Team models", `Elo, Dixon-Coles MLE (goals & xG), Monte Carlo, walk-forward backtest — fitted in ${t.ratings}s.`],
          ["scouting.py · network.py", "5 · Player & graph models", `Per-90 features → PCA → k-means++ (silhouette-selected k) in ${t.scouting}s; PageRank &amp; Brandes per match on demand.`],
        ].map(([tag, h, p]) => `<div class="stage"><span class="tag">${tag}</span><h4>${h}</h4><p>${p}</p></div>`).join("")}
      </div>
    </div>
    <div class="grid g-2">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">API explorer</h3><div class="card-sub">Click an endpoint to call it live · full schema at <a href="/docs" target="_blank">/docs</a></div></div></div>
        <div>${ENDPOINTS.map(([m, path, desc]) => `<div class="endpoint" data-m="${m}" data-p="${path}" style="cursor:${m === "WS" ? "default" : "pointer"}">
          <span class="method ${m.toLowerCase()}">${m}</span><div><div class="mono" style="font-size:12px">${path}</div><div class="muted" style="font-size:11.5px">${desc}</div></div></div>`).join("")}</div>
      </div>
      <div class="card" style="align-self:start;position:sticky;top:90px">
        <div class="card-head"><div><h3 class="card-title">Response</h3><div class="card-sub" id="resp-meta">Select an endpoint</div></div></div>
        <pre id="resp" class="mono" style="margin:0;font-size:11.5px;max-height:560px;overflow:auto;background:var(--surface-2);padding:12px;border-radius:10px;white-space:pre-wrap;word-break:break-word">{}</pre>
      </div>
    </div>
    <div class="grid g-3">
      <div class="card"><h3 class="card-title">Backend</h3><div class="card-note">
        <b>Python 3 · FastAPI · Uvicorn</b> (ASGI, async WebSockets) · <b>NumPy</b> for every model (IRLS, SVD-PCA, k-means++, value iteration, PageRank, Brandes) ·
        <b>SciPy</b> L-BFGS-B for Dixon-Coles MLE · <b>pandas</b> for feature engineering · <b>SQLite</b> event store · Pydantic request validation ·
        GZip, CORS, per-IP sliding-window rate limiting and an <code>X-Process-Time-ms</code> header via middleware · pytest suite (unit + API + WebSocket).</div></div>
      <div class="card"><h3 class="card-title">Frontend</h3><div class="card-note">
        <b>Vanilla JavaScript (ES modules)</b>, no build step · <b>D3.js v7</b> for every chart (pitch rendering in native 120×80 StatsBomb units, contours, stacked areas, symbol scatter) ·
        hash router with lazily-imported views · theme tokens in CSS custom properties (light/dark) · ResizeObserver-driven responsive redraws · native WebSocket client.</div></div>
      <div class="card"><h3 class="card-title">${real ? "Real data, honest validation" : "Why simulate the data?"}</h3><div class="card-note">${real
        ? `Every match, player and event comes from <b>${o.source.competition}</b> (StatsBomb open data). Because the season is complete, every forecast is
           tested out-of-sample: models are fitted only on matches before the ones they predict, and the xG model is benchmarked against StatsBomb's own.
           A synthetic league with known ground truth is still available (<code>PITCHIQ_SOURCE=synthetic</code>) to prove the estimators recover true parameters.`
        : `Real event data is licensed and has no ground truth. The simulator knows each team's true strength and every shot's true probability,
        so the app can <b>prove</b> its models work (xG within 1% of the oracle log-loss; ratings correlate with hidden strength).`}
        The schema mirrors StatsBomb, so open data can be plugged into the same pipeline.</div></div>
    </div>`;

  el.querySelectorAll(".endpoint").forEach((e) => e.addEventListener("click", async () => {
    const m = e.dataset.m, path = e.dataset.p;
    if (m === "WS") return;
    const pre = el.querySelector("#resp"), meta = el.querySelector("#resp-meta");
    pre.textContent = "…";
    const t0 = performance.now();
    try {
      const body = m === "POST" ? [{ x: 108, y: 40 }, { x: 100, y: 25, header: true }] : undefined;
      const r = await fetch(path, { method: m, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
      const txt = await r.text();
      meta.textContent = `${m} ${path} → ${r.status} · ${(performance.now() - t0).toFixed(0)} ms · server ${r.headers.get("X-Process-Time-ms")} ms · ${(txt.length / 1024).toFixed(1)} KB`;
      const json = JSON.parse(txt);
      const s = JSON.stringify(json, null, 2);
      pre.textContent = s.length > 6000 ? `${s.slice(0, 6000)}\n… (truncated)` : s;
    } catch (err) { pre.textContent = String(err); }
  }));
}
