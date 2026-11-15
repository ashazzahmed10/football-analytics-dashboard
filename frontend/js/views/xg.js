/* global d3 */
import { api } from "../api.js";
import { css, fmt, hBars, hideTip, legendHTML, lineChart, loading, pitch, responsive, seqScale, showTip, tipRows } from "../charts.js";

export async function render(el) {
  loading(el, "Loading model report…");
  const rep = await api("/api/models/xg");
  const M = rep.metrics;
  el.innerHTML = `
    <div class="grid g-21">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Interactive shot simulator</h3>
          <div class="card-sub">Click anywhere in the attacking half to score a shot with the live model (POST /api/models/xg/predict)</div></div>
          <div class="controls">
            <div class="seg" id="sit">${["open_play", "counter", "corner", "free_kick"].map((s, i) => `<button data-v="${s}" class="${i ? "" : "on"}">${s.replace("_", " ")}</button>`).join("")}</div>
            <div class="seg" id="body"><button data-v="foot" class="on">Foot</button><button data-v="head">Header</button></div>
          </div></div>
        <div class="chart half-pitch" id="surface" style="cursor:crosshair"></div>
        <div class="controls" style="justify-content:space-between;margin-top:10px">
          <div id="surface-legend"></div><span class="muted" style="font-size:12px">Contours at 5%, 10%, 20%, 35%, 50%</span>
        </div>
      </div>
      <div class="grid" style="align-content:start">
        <div class="card tile"><div class="label">Selected shot</div><div class="value" id="shot-xg">–</div><div class="hint" id="shot-desc">Click the pitch</div></div>
        <div class="card"><h3 class="card-title" style="margin-bottom:10px">Held-out performance</h3>
          <table class="data"><thead><tr><th>Model</th><th>Log-loss</th><th>Brier</th><th>AUC</th></tr></thead><tbody>
          ${[["PitchIQ xG", M.model], ["Distance only", M.distance_only], ["Base rate", M.base_rate], ["Oracle (true p)", M.oracle]].map(([n, m], i) =>
            `<tr><td class="${i === 0 ? "strong" : ""}">${n}</td><td>${m.log_loss.toFixed(4)}</td><td>${m.brier.toFixed(4)}</td><td>${m.auc.toFixed(3)}</td></tr>`).join("")}
          </tbody></table>
          <div class="card-note">Trained on <b>${fmt.int(rep.n_train)}</b> shots, tested on <b>${fmt.int(rep.n_test)}</b>. Correlation with the simulator's hidden true probabilities:
            <b>r = ${rep.corr_with_truth.toFixed(3)}</b>. The model's log-loss is within ${((M.model.log_loss / M.oracle.log_loss - 1) * 100).toFixed(1)}% of the theoretical optimum (oracle).</div>
        </div>
      </div>
    </div>
    <div class="grid g-3">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Calibration</h3><div class="card-sub">Predicted vs observed goal rate (deciles)</div></div></div>
        <div class="chart" id="calib"></div><div class="card-note">Points on the diagonal = probabilities you can trust at face value.</div></div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">ROC curve</h3><div class="card-sub">AUC ${M.model.auc.toFixed(3)} on test set</div></div></div>
        <div class="chart" id="roc"></div></div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">IRLS convergence</h3><div class="card-sub">Training log-loss per Newton step</div></div></div>
        <div class="chart" id="conv"></div><div class="card-note">Newton-Raphson converges quadratically — <b>${rep.iterations} iterations</b> vs. thousands for gradient descent.</div></div>
    </div>
    <div class="card">
      <div class="card-head"><div><h3 class="card-title">Feature effects</h3><div class="card-sub">Wald z-statistic (β / SE) on standardised features — |z| > 2 is significant</div></div></div>
      <div class="grid g-2"><div class="chart" id="coef"></div>
        <div class="explain" style="margin-top:0">
          <b>How it works.</b> Each shot becomes a feature vector: distance, goal-mouth angle (the angle between the posts as seen from the ball),
          their interaction, log-distance, and flags for header / counter-attack / corner / direct free-kick.
          The model is <code>p = σ(β·x)</code>, fitted by <b>Iteratively Re-weighted Least Squares</b>: each step solves
          <code>(XᵀWX + λI)Δ = Xᵀ(y − p) − λβ</code> with <code>W = diag(p(1−p))</code>. The inverse Hessian at the optimum gives
          standard errors for free. Penalties use the empirical conversion rate (${fmt.pct(rep.penalty_xg, 1)}).
        </div></div>
    </div>`;

  let situation = "open_play", header = false, marker = null;
  const surfEl = el.querySelector("#surface");
  const drawSurface = async () => {
    const s = await api(`/api/models/xg/surface?header=${header}&situation=${situation}`);
    surfEl.innerHTML = "";
    const { svg, g } = pitch(surfEl, { half: true });
    const nx = s.x.length, ny = s.y.length;
    const scale = seqScale([0, 0.6]);
    const cw = (s.x[1] - s.x[0]), ch = (s.y[1] - s.y[0]);
    const cells = s.z.flatMap((row, j) => row.map((v, i) => ({ x: s.x[i], y: s.y[j], v })));
    g.insert("g", ".pitch-markings").selectAll("rect").data(cells).join("rect")
      .attr("x", (d) => d.x - cw / 2).attr("y", (d) => d.y - ch / 2).attr("width", cw + 0.05).attr("height", ch + 0.05)
      .attr("fill", (d) => scale(Math.min(d.v, 0.6)));
    const contours = d3.contours().size([nx, ny]).thresholds([0.05, 0.1, 0.2, 0.35, 0.5])(s.z.flat());
    const tx = d3.scaleLinear().domain([0, nx - 1]).range([s.x[0], s.x[nx - 1]]);
    const ty = d3.scaleLinear().domain([0, ny - 1]).range([s.y[0], s.y[ny - 1]]);
    const path = d3.geoPath(d3.geoTransform({ point(x, y) { this.stream.point(tx(x - 0.5), ty(y - 0.5)); } }));
    g.append("g").selectAll("path").data(contours).join("path").attr("d", path).attr("fill", "none")
      .attr("stroke", "#fff").attr("stroke-opacity", 0.55).attr("stroke-width", 0.25).attr("pointer-events", "none");
    // hover readout
    g.select(".pitch-bg").attr("pointer-events", "none");
    const hit = g.append("rect").attr("x", 60).attr("y", 0).attr("width", 60).attr("height", 80).attr("fill", "transparent");
    const toPitch = (event) => { const pt = d3.pointer(event, g.node()); return [Math.max(60, Math.min(119.5, pt[0])), Math.max(0.5, Math.min(79.5, pt[1]))]; };
    hit.on("mousemove", (event) => {
      const [x, y] = toPitch(event);
      const i = Math.round((x - s.x[0]) / cw), j = Math.round((y - s.y[0]) / ch);
      const v = s.z[Math.max(0, Math.min(ny - 1, j))][Math.max(0, Math.min(nx - 1, i))];
      showTip(event, tipRows("Model surface", [["xG", v.toFixed(3)], ["Distance", `${Math.hypot(120 - x, 40 - y).toFixed(1)} yd`]]));
    }).on("mouseleave", hideTip).on("click", async (event) => {
      const [x, y] = toPitch(event);
      const [res] = await api("/api/models/xg/predict", { method: "POST", body: [{ x, y, header, situation }] });
      marker = { x, y };
      placeMarker(g, marker, res.xg);
      el.querySelector("#shot-xg").textContent = res.xg.toFixed(3);
      el.querySelector("#shot-desc").textContent = `${Math.hypot(120 - x, 40 - y).toFixed(1)} yd · ${header ? "header" : "foot"} · ${situation.replace("_", " ")} — scores ~${Math.round(res.xg * 100)} in 100`;
    });
    if (marker) {
      const [res] = await api("/api/models/xg/predict", { method: "POST", body: [{ ...marker, header, situation }] });
      placeMarker(g, marker, res.xg);
      el.querySelector("#shot-xg").textContent = res.xg.toFixed(3);
    }
    el.querySelector("#surface-legend").innerHTML = gradientLegend(scale, [0, 0.6], "xG");
  };
  const seg = (id, fn) => el.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    el.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b)); fn(b.dataset.v); drawSurface();
  }));
  seg("sit", (v) => { situation = v; });
  seg("body", (v) => { header = v === "head"; });
  drawSurface();

  const ros = [];
  const cal = el.querySelector("#calib");
  ros.push(responsive(cal, () => {
    const mx = Math.max(...rep.calibration.map((c) => Math.max(c.predicted, c.observed))) * 1.05;
    lineChart(cal, { series: [{ name: "Perfect", color: css("--axis"), dash: "4 4", width: 1.5, values: [{ x: 0, y: 0 }, { x: mx, y: mx }] },
      { name: "Model", color: css("--series-1"), values: rep.calibration.map((c) => ({ x: c.predicted, y: c.observed })) }],
    xDomain: [0, mx], yDomain: [0, mx], height: 240, xLabel: "Predicted", yFormat: (v) => v.toFixed(2), xFormat: (v) => v.toFixed(2), curve: d3.curveLinear,
    markers: (svg, x, y) => svg.selectAll(".c").data(rep.calibration).join("circle").attr("cx", (c) => x(c.predicted)).attr("cy", (c) => y(c.observed))
      .attr("r", 4.5).attr("fill", css("--series-1")).attr("stroke", css("--surface-1")).attr("stroke-width", 2) });
  }));
  const roc = el.querySelector("#roc");
  ros.push(responsive(roc, () => lineChart(roc, { series: [{ name: "Random", color: css("--axis"), dash: "4 4", width: 1.5, values: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
    { name: "Model", color: css("--series-1"), values: rep.roc.map((r) => ({ x: r.fpr, y: r.tpr })) }],
  xDomain: [0, 1], yDomain: [0, 1], height: 240, xLabel: "False positive rate", yFormat: (v) => v.toFixed(1), xFormat: (v) => v.toFixed(1), curve: d3.curveLinear, area: true })));
  const conv = el.querySelector("#conv");
  ros.push(responsive(conv, () => lineChart(conv, { series: [{ name: "Log-loss", color: css("--series-3"), values: [{ x: 0, y: Math.log(2) }, ...rep.convergence.map((v, i) => ({ x: i + 1, y: v }))] }],
    height: 240, xLabel: "Iteration", yFormat: (v) => v.toFixed(3), curve: d3.curveLinear,
    markers: (svg, x, y) => svg.selectAll(".p").data([Math.log(2), ...rep.convergence]).join("circle").attr("cx", (v, i) => x(i)).attr("cy", (v) => y(v)).attr("r", 4).attr("fill", css("--series-3")).attr("stroke", css("--surface-1")).attr("stroke-width", 2) })));
  const coef = el.querySelector("#coef");
  const labels = { distance: "Distance", angle: "Goal angle", distance_x_angle: "Distance × angle", log_distance: "log(distance)", header: "Header", counter: "Counter-attack", corner: "From corner", free_kick: "Direct free kick" };
  ros.push(responsive(coef, () => hBars(coef, { data: [...rep.coefficients].sort((a, b) => b.z - a.z), value: (d) => d.z, label: (d) => labels[d.feature] || d.feature,
    diverging: true, labelW: 130, rowH: 28, color: (d) => (d.z >= 0 ? css("--series-1") : css("--series-2")), format: (v) => v.toFixed(1) })));
  return () => ros.forEach((r) => r.disconnect());
}

function placeMarker(g, m, xg) {
  g.selectAll(".marker").remove();
  const mk = g.append("g").attr("class", "marker").attr("pointer-events", "none");
  mk.append("line").attr("x1", m.x).attr("y1", m.y).attr("x2", 120).attr("y2", 36).attr("stroke", "#fff").attr("stroke-width", 0.2).attr("stroke-dasharray", "0.8 0.6");
  mk.append("line").attr("x1", m.x).attr("y1", m.y).attr("x2", 120).attr("y2", 44).attr("stroke", "#fff").attr("stroke-width", 0.2).attr("stroke-dasharray", "0.8 0.6");
  mk.append("circle").attr("cx", m.x).attr("cy", m.y).attr("r", 1.4).attr("fill", css("--series-2")).attr("stroke", "#fff").attr("stroke-width", 0.4);
  mk.append("text").attr("x", m.x).attr("y", m.y - 2.6).attr("text-anchor", "middle").attr("font-size", 2.8).attr("font-weight", 700).attr("fill", "#fff")
    .attr("paint-order", "stroke").attr("stroke", "rgba(0,0,0,.6)").attr("stroke-width", 0.5).text(xg.toFixed(2));
}

export function gradientLegend(scale, [lo, hi], label, f = (v) => v.toFixed(2)) {
  const stops = d3.range(0, 1.01, 0.1).map((t) => `${scale(lo + t * (hi - lo))} ${t * 100}%`).join(",");
  return `<div class="legend" style="align-items:center;gap:8px"><span>${label}</span><span>${f(lo)}</span>
    <span style="width:140px;height:10px;border-radius:5px;background:linear-gradient(90deg,${stops})"></span><span>${f(hi)}+</span></div>`;
}
