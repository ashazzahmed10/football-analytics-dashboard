/* global d3 */
import { api } from "../api.js";
import { crest, css, fmt, hideTip, loading, seqScale, showTip, tipRows } from "../charts.js";
import { gradientLegend } from "./xg.js";

export async function render(el) {
  el.innerHTML = `
    <div class="card"><div class="controls" style="justify-content:space-between">
      <div class="controls">
        <div class="seg" id="nsims"><button data-v="1000">1k sims</button><button data-v="10000" class="on">10k sims</button><button data-v="50000">50k sims</button></div>
        <select class="select" id="model"><option value="dixon_coles">Dixon-Coles (goals)</option><option value="xg_poisson">Poisson (xG-based)</option></select>
        <button class="btn primary" id="run">Run simulation</button>
        <button class="btn" id="reseed" title="Re-run with a new random seed">🎲 New seed</button>
      </div>
      <span class="chip" id="sim-info">–</span>
    </div></div>
    <div id="out" class="grid"></div>`;
  let n = 10000, seed = 0;
  el.querySelectorAll("#nsims button").forEach((b) => b.addEventListener("click", () => {
    n = +b.dataset.v; el.querySelectorAll("#nsims button").forEach((x) => x.classList.toggle("on", x === b)); run();
  }));
  const run = async () => {
    const out = el.querySelector("#out");
    const model = el.querySelector("#model").value;
    loading(out, `Simulating ${fmt.int(n)} seasons…`);
    const t0 = performance.now();
    const s = await api(`/api/simulate/season?n=${n}&model=${model}&seed=${seed}`);
    el.querySelector("#sim-info").textContent = `${fmt.int(s.n_sims)} seasons × ${s.remaining_fixtures} fixtures = ${fmt.int(s.n_sims * s.remaining_fixtures)} matches · server ${s.seconds}s · round-trip ${((performance.now() - t0) / 1000).toFixed(2)}s`;
    draw(out, s);
  };
  el.querySelector("#run").addEventListener("click", run);
  el.querySelector("#model").addEventListener("change", run);
  el.querySelector("#reseed").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e6); run(); });
  run();
}

function draw(out, s) {
  const teams = s.teams;
  const bar = (v, color) => `<div class="bar-cell"><span>${v >= 0.001 ? fmt.pct(v, v < 0.1 ? 1 : 0) : "–"}</span><div class="track"><div class="fill" style="width:${v * 100}%;background:${color}"></div></div></div>`;
  out.innerHTML = `
    <div class="grid">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Projected final table</h3><div class="card-sub">Sorted by expected finishing position</div></div></div>
        <div class="table-wrap"><table class="data"><thead><tr><th>Team</th><th>Pts now</th><th>Proj. pts (10–90%)</th><th>Title</th><th>Top 4</th><th>Relegated</th></tr></thead>
        <tbody>${teams.map((t) => `<tr>
          <td><div class="team-cell">${crest(t.short)}<span>${t.name}</span></div></td><td>${t.current_points}</td>
          <td class="strong">${t.expected_points.toFixed(1)} <span class="muted" style="font-weight:400">(${Math.round(t.points_p10)}–${Math.round(t.points_p90)})</span></td>
          <td>${bar(t.title, css("--series-1"))}</td><td>${bar(t.top4, css("--series-3"))}</td><td>${bar(t.relegation, css("--critical"))}</td></tr>`).join("")}</tbody></table></div></div>
      <div class="card" style="max-width:900px;width:100%;justify-self:center"><div class="card-head"><div><h3 class="card-title">Finishing-position probability</h3><div class="card-sub">Each row sums to 100%</div></div><div id="leg"></div></div>
        <div class="chart" id="heat"></div></div>
    </div>
    <div class="card"><div class="explain" style="margin-top:0"><b>Vectorised Monte Carlo.</b> For each of the ${s.remaining_fixtures} remaining fixtures,
      the model's full scoreline distribution (11×11 Dixon-Coles matrix) is sampled <b>${fmt.int(s.n_sims)}</b> times at once with NumPy — no Python loop over simulations.
      Points, goal difference and goals scored are accumulated as <code>(n_sims × 20)</code> matrices, then ranked with a single <code>argsort</code> using the official tie-breakers.
      Monte Carlo error on a 50% probability at 10k sims is ±${(100 * 1.96 * Math.sqrt(0.25 / s.n_sims)).toFixed(1)} pp (95% CI).</div></div>`;

  const container = out.querySelector("#heat");
  const W = Math.max(container.clientWidth, 340), labelW = 44, top = 20, n = teams.length;
  const cell = (W - labelW) / n, H = top + cell * n;
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const scale = seqScale([0, 0.6]);
  out.querySelector("#leg").innerHTML = gradientLegend(scale, [0, 0.6], "P", (v) => fmt.pct(v));
  const cells = teams.flatMap((t, i) => t.positions.map((v, j) => ({ t, i, j, v })));
  svg.selectAll("rect").data(cells).join("rect").attr("x", (c) => labelW + c.j * cell + 1).attr("y", (c) => top + c.i * cell + 1)
    .attr("width", cell - 2).attr("height", cell - 2).attr("rx", 2).attr("fill", (c) => (c.v < 0.0005 ? css("--surface-2") : scale(Math.min(c.v, 0.6))))
    .on("mousemove", (e, c) => showTip(e, tipRows(c.t.name, [["Finishes", `${c.j + 1}${["st", "nd", "rd"][c.j] || "th"}`], ["Probability", fmt.pct(c.v, 1)]])))
    .on("mouseleave", hideTip);
  svg.selectAll("text.t").data(teams).join("text").attr("x", labelW - 6).attr("y", (t, i) => top + i * cell + cell / 2).attr("dy", "0.35em")
    .attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", css("--text-secondary")).text((t) => t.short);
  svg.selectAll("text.p").data(d3.range(n)).join("text").attr("x", (j) => labelW + j * cell + cell / 2).attr("y", top - 7)
    .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", css("--text-muted")).text((j) => j + 1);
}
