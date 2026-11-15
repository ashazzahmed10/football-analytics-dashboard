/* global d3 */
import { api } from "../api.js";
import { css, fmt, hBars, hideTip, legendHTML, lineChart, loading, responsive, showTip, tipRows } from "../charts.js";

const NAMES = { ensemble: "Ensemble", xg_poisson: "xG-Poisson", dixon_coles: "Dixon-Coles", elo: "Elo", uniform: "Uniform (⅓ each)" };

export async function render(el) {
  loading(el, "Running walk-forward backtest (refits every model for each matchday)…");
  const [bt, rt] = await Promise.all([api("/api/models/backtest"), api("/api/league/ratings")]);
  const S = bt.summary;
  const order = Object.keys(S).sort((a, b) => S[a].rps - S[b].rps);
  const colors = { ensemble: css("--series-1"), xg_poisson: css("--series-3"), dixon_coles: css("--series-2"), elo: css("--series-7"), uniform: css("--neutral") };
  el.innerHTML = `
    <div class="grid g-2">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Walk-forward backtest</h3>
        <div class="card-sub">For every matchday k ≥ ${bt.start_matchday}: fit on matchdays &lt; k, forecast k. No look-ahead.</div></div></div>
        <table class="data"><thead><tr><th>Model</th><th>RPS ↓</th><th>Log-loss ↓</th><th>Accuracy ↑</th><th>Skill vs uniform</th></tr></thead><tbody>
        ${order.map((k, i) => `<tr><td class="${i === 0 ? "strong" : ""}"><span class="team-dot" style="background:${colors[k]};margin-right:8px;vertical-align:-1px"></span>${NAMES[k]}</td>
          <td>${S[k].rps.toFixed(4)}</td><td>${S[k].log_loss.toFixed(3)}</td><td>${fmt.pct(S[k].accuracy, 1)}</td>
          <td>${k === "uniform" ? "–" : fmt.pct(1 - S[k].rps / S.uniform.rps, 1)}</td></tr>`).join("")}</tbody></table>
        <div class="card-note"><b>Ranked Probability Score</b> is the standard metric for football forecasts: it rewards putting probability mass <i>near</i> the
          actual result (a draw is closer to a home win than an away win is). Evaluated on ${S.elo.n} out-of-sample matches.</div></div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">Cumulative RPS by matchday</h3><div class="card-sub">Lower is better · models improve as data accumulates</div></div></div>
        <div class="chart" id="cum"></div>${legendHTML(order.map((k) => ({ label: NAMES[k], color: colors[k] })), { line: true })}</div>
    </div>
    <div class="grid g-2">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Ground-truth recovery</h3>
        <div class="card-sub">Fitted attack rating vs the simulator's hidden true attack strength</div></div><div class="seg" id="dim"><button data-k="attack" class="on">Attack</button><button data-k="defence">Defence</button></div></div>
        <div class="chart" id="truth"></div>${legendHTML([{ label: "Goals model", color: css("--series-2") }, { label: "xG model", color: css("--series-3") }])}
        <div class="card-note" id="truth-note"></div></div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">Fitted team strengths (Dixon-Coles)</h3><div class="card-sub">attack + defence · log-goal scale</div></div></div>
        <div class="chart" id="strength"></div>
        <div class="kv" style="margin-top:14px">
          <span class="k">Home advantage (log scale)</span><span class="v">${rt.params.home_advantage.toFixed(3)} → ×${Math.exp(rt.params.home_advantage).toFixed(2)} goals</span>
          <span class="k">Low-score correction ρ</span><span class="v">${rt.params.rho.toFixed(3)}</span>
          <span class="k">Time-decay ξ</span><span class="v">${rt.params.xi}</span>
          <span class="k">L-BFGS-B iterations</span><span class="v">${rt.params.iterations} (${rt.params.converged ? "converged" : "not converged"})</span>
        </div></div>
    </div>`;

  const ros = [];
  const cum = el.querySelector("#cum");
  const series = order.map((k) => {
    let s = 0;
    return { name: NAMES[k], color: colors[k], values: bt.per_matchday.map((m, i) => { s += m[k]; return { x: m.matchday, y: s / (i + 1) }; }) };
  });
  ros.push(responsive(cum, () => lineChart(cum, { series, height: 260, xLabel: "Matchday", yFormat: (v) => v.toFixed(3), curve: d3.curveLinear })));

  let dim = "attack";
  const tEl = el.querySelector("#truth");
  const drawTruth = () => {
    const r = rt.truth_recovery;
    el.querySelector("#truth-note").innerHTML = `Pearson r — goals model: <b>${r.goals_model[`${dim}_r`]}</b> · xG model: <b>${r.xg_model[`${dim}_r`]}</b>.
      Because xG strips out finishing luck, it recovers true quality better from the same number of matches — the core argument for xG-based ratings.`;
    scatter(tEl, rt.teams, dim);
  };
  el.querySelectorAll("#dim button").forEach((b) => b.addEventListener("click", () => {
    dim = b.dataset.k; el.querySelectorAll("#dim button").forEach((x) => x.classList.toggle("on", x === b)); drawTruth();
  }));
  ros.push(responsive(tEl, drawTruth));
  const st = el.querySelector("#strength");
  ros.push(responsive(st, () => hBars(st, { data: [...rt.teams].sort((a, b) => b.attack + b.defence - a.attack - a.defence), value: (t) => t.attack + t.defence,
    label: (t) => t.name, labelW: 150, rowH: 21, diverging: true, color: (t) => (t.attack + t.defence >= 0 ? css("--series-1") : css("--series-2")), format: (v) => fmt.signed(v, 2) })));
  return () => ros.forEach((r) => r.disconnect());
}

function scatter(container, teams, dim) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const W = Math.max(container.clientWidth, 300), H = 300, m = { t: 12, r: 14, b: 38, l: 46 };
  const svg = el.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const tk = `true_${dim}`, gk = dim, xk = `xg_${dim}`;
  const xs = teams.map((t) => t[tk]), ys = teams.flatMap((t) => [t[gk], t[xk]]);
  const x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(ys)).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(6)).select(".domain").remove();
  svg.append("text").attr("class", "axis-title").attr("x", W - m.r).attr("y", H - 6).attr("text-anchor", "end").text(`True ${dim} (hidden)`);
  svg.append("text").attr("class", "axis-title").attr("x", m.l - 40).attr("y", m.t - 2).text("Fitted");
  for (const [key, color, shape] of [[gk, css("--series-2"), d3.symbolCircle], [xk, css("--series-3"), d3.symbolDiamond]]) {
    const reg = linreg(xs, teams.map((t) => t[key]));
    svg.append("line").attr("x1", x(x.domain()[0])).attr("x2", x(x.domain()[1])).attr("y1", y(reg(x.domain()[0]))).attr("y2", y(reg(x.domain()[1])))
      .attr("stroke", color).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 4").attr("opacity", 0.8);
    svg.selectAll(null).data(teams).join("path").attr("d", d3.symbol(shape, 70)).attr("transform", (t) => `translate(${x(t[tk])},${y(t[key])})`)
      .attr("fill", color).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5)
      .on("mousemove", (e, t) => showTip(e, tipRows(t.name, [["True", t[tk].toFixed(3)], ["Goals model", t[gk].toFixed(3)], ["xG model", t[xk].toFixed(3)]])))
      .on("mouseleave", hideTip);
  }
}

function linreg(xs, ys) {
  const mx = d3.mean(xs), my = d3.mean(ys);
  const b = d3.sum(xs.map((x, i) => (x - mx) * (ys[i] - my))) / d3.sum(xs.map((x) => (x - mx) ** 2));
  return (x) => my + b * (x - mx);
}
