/* global d3 */
import { api } from "../api.js";
import { css, hBars, lineChart, loading, pitch, pitchGrid, responsive, seqScale, showTip, tipRows } from "../charts.js";
import { gradientLegend } from "./xg.js";

export async function render(el) {
  loading(el, "Solving the Markov chain…");
  const d = await api("/api/models/xt");
  const layers = { grid: "Expected threat", shot_prob: "P(shoot)", goal_prob: "P(goal | shot)", move_prob: "P(move)" };
  el.innerHTML = `
    <div class="card">
      <div class="card-head"><div><h3 class="card-title">Expected Threat surface</h3>
        <div class="card-sub">${d.nx}×${d.ny} grid · value of possessing the ball in each zone (probability of scoring within the next few actions)</div></div>
        <div class="seg" id="layer">${Object.entries(layers).map(([k, v], i) => `<button data-k="${k}" class="${i ? "" : "on"}">${v}</button>`).join("")}</div></div>
      <div class="chart" id="xtmap"></div>
      <div class="controls" style="margin-top:12px;justify-content:space-between">
        <div class="controls"><button class="btn" id="play">▶ Animate value iteration</button>
          <input type="range" id="iter" min="0" max="${d.snapshots.length - 1}" value="${d.snapshots.length - 1}" style="width:200px" aria-label="Iteration" />
          <span class="mono muted" id="iter-label">converged</span></div>
        <div id="xt-legend"></div>
      </div>
      <div class="explain"><b>Bellman equation:</b> <code>xT(z) = s(z)·g(z) + m(z)·Σ T(z→z′)·xT(z′)</code>. Shoot now and score with probability <i>g</i>,
        or move the ball to another zone and inherit <i>its</i> threat. Starting from xT = 0, each iteration lets value propagate one more action back up the pitch —
        converged in <b>${d.iterations} iterations</b> (max Δ &lt; 10⁻⁶). The transition matrix T is ${(d.nx * d.ny)}×${(d.nx * d.ny)}, estimated from every successful pass and carry.</div>
    </div>
    <div class="grid g-2">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Convergence</h3><div class="card-sub">max |xTₖ − xTₖ₋₁| (log scale)</div></div></div><div class="chart" id="deltas"></div></div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">xT added per 90 — leaders</h3><div class="card-sub">Σ xT(end) − xT(start) over successful actions · 600+ min</div></div></div><div class="chart" id="leaders"></div></div>
    </div>`;

  let layer = "grid", snap = d.snapshots.length - 1, timer = null;
  const mapEl = el.querySelector("#xtmap");
  const draw = () => {
    mapEl.innerHTML = "";
    const { g } = pitch(mapEl);
    const grid = layer === "grid" ? d.snapshots[snap] : d[layer];
    const max = layer === "grid" ? d3.max(d.grid.flat()) : d3.max(grid.flat());
    const scale = seqScale([0, max]);
    const cw = 120 / d.nx, ch = 80 / d.ny;
    pitchGrid(g, grid, { scale, onHover: (e, c) => showTip(e, tipRows(`Zone (${c.i}, ${c.j})`, [[layers[layer], c.v.toFixed(4)], ["x", `${(c.i * cw).toFixed(0)}–${((c.i + 1) * cw).toFixed(0)}`]])) });
    g.append("g").attr("pointer-events", "none").selectAll("text").data(grid.flatMap((row, j) => row.map((v, i) => ({ v, i, j })))).join("text")
      .attr("x", (c) => c.i * cw + cw / 2).attr("y", (c) => c.j * ch + ch / 2).attr("dy", "0.35em").attr("text-anchor", "middle")
      .attr("font-size", 1.9).attr("fill", (c) => (c.v > max * 0.45 ? (document.documentElement.dataset.theme === "light" ? "#fff" : "#0b0b0b") : css("--text-secondary")))
      .attr("opacity", 0.85).text((c) => (c.v >= 0.001 ? (layer === "grid" ? (c.v * 100).toFixed(1) : c.v.toFixed(2)) : ""));
    el.querySelector("#xt-legend").innerHTML = gradientLegend(scale, [0, max], layers[layer] + (layer === "grid" ? " (labels in %)" : ""), (v) => v.toFixed(2));
    el.querySelector("#iter-label").textContent = layer !== "grid" ? "—" : snap === d.snapshots.length - 1 ? `converged (${d.iterations} iters)` : `iteration ${snap + 1}`;
  };
  el.querySelectorAll("#layer button").forEach((b) => b.addEventListener("click", () => {
    layer = b.dataset.k; el.querySelectorAll("#layer button").forEach((x) => x.classList.toggle("on", x === b)); draw();
  }));
  const slider = el.querySelector("#iter");
  slider.addEventListener("input", () => { snap = +slider.value; layer = "grid"; draw(); });
  el.querySelector("#play").addEventListener("click", () => {
    clearInterval(timer);
    layer = "grid";
    el.querySelectorAll("#layer button").forEach((x) => x.classList.toggle("on", x.dataset.k === "grid"));
    snap = 0;
    timer = setInterval(() => { slider.value = snap; draw(); if (++snap >= d.snapshots.length) clearInterval(timer); }, 550);
  });
  draw();

  const ros = [];
  const de = el.querySelector("#deltas");
  ros.push(responsive(de, () => {
    const { svg } = lineChart(de, { series: [{ name: "max Δ", color: css("--series-3"), values: d.deltas.map((v, i) => ({ x: i + 1, y: Math.log10(v) })) }],
      height: 250, xLabel: "Iteration", yFormat: (v) => `1e${Math.round(v)}`, curve: d3.curveLinear });
    return svg;
  }));
  const le = el.querySelector("#leaders");
  ros.push(responsive(le, () => hBars(le, { data: d.leaders.slice(0, 12), value: (r) => r.xt_p90, label: (r) => `${r.name} (${r.position})`, labelW: 170, color: css("--series-1"), format: (v) => v.toFixed(3) })));
  return () => { clearInterval(timer); ros.forEach((r) => r.disconnect()); };
}
