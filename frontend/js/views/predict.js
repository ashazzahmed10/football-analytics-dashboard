/* global d3 */
import { api } from "../api.js";
import { AWAY, HOME, css, fmt, hideTip, loading, seqScale, showTip, tipRows } from "../charts.js";

export async function render(el) {
  const [teams, table] = await Promise.all([api("/api/teams"), api("/api/league/table")]);
  const opts = (sel) => teams.map((t) => `<option value="${t.id}" ${t.id === sel ? "selected" : ""}>${t.name}</option>`).join("");
  let home = table[0].team_id, away = table[1].team_id;
  el.innerHTML = `
    <div class="card"><div class="controls">
      <span class="team-dot" style="background:${HOME()}"></span><select class="select" id="home">${opts(home)}</select>
      <span class="muted">vs</span>
      <span class="team-dot" style="background:${AWAY()}"></span><select class="select" id="away">${opts(away)}</select>
      <button class="btn" id="swap">⇄ Swap venue</button>
    </div></div>
    <div id="out" class="grid"></div>`;
  const run = async () => {
    const out = el.querySelector("#out");
    if (home === away) { out.innerHTML = `<div class="card empty">Pick two different teams.</div>`; return; }
    loading(out, "Computing score matrix…");
    draw(out, await api(`/api/predict?home=${home}&away=${away}`));
  };
  el.querySelector("#home").addEventListener("change", (e) => { home = +e.target.value; run(); });
  el.querySelector("#away").addEventListener("change", (e) => { away = +e.target.value; run(); });
  el.querySelector("#swap").addEventListener("click", () => {
    [home, away] = [away, home];
    el.querySelector("#home").value = home; el.querySelector("#away").value = away; run();
  });
  run();
}

function draw(out, p) {
  const names = { dixon_coles: "Dixon-Coles (goals)", xg_poisson: "Poisson (xG-based)", elo: "Elo", ensemble: "Ensemble (mean)" };
  out.innerHTML = `
    <div class="grid g-2">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Outcome probabilities</h3><div class="card-sub">Three independent models and their average</div></div></div>
        <div id="probs" style="display:grid;gap:14px"></div>
        <div class="kv" style="margin-top:18px">
          <span class="k">Expected goals (Dixon-Coles λ)</span><span class="v">${p.expected_goals.home.toFixed(2)} – ${p.expected_goals.away.toFixed(2)}</span>
          <span class="k">Both teams score</span><span class="v">${fmt.pct(p.btts, 1)}</span>
          <span class="k">Over 2.5 goals</span><span class="v">${fmt.pct(p.over_2_5, 1)}</span>
          <span class="k">${p.home.short} clean sheet</span><span class="v">${fmt.pct(p.clean_sheet.home, 1)}</span>
          <span class="k">${p.away.short} clean sheet</span><span class="v">${fmt.pct(p.clean_sheet.away, 1)}</span>
          <span class="k">Fair odds (ensemble)</span><span class="v">${["home", "draw", "away"].map((k) => (1 / p.models.ensemble[k]).toFixed(2)).join(" / ")}</span>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Scoreline probability matrix</h3><div class="card-sub">Rows: ${p.home.name} goals · Columns: ${p.away.name} goals</div></div></div>
        <div class="chart" id="matrix"></div>
        <div class="card-note">Most likely: ${p.top_scores.slice(0, 4).map((s) => `<b>${s.home}–${s.away}</b> (${fmt.pct(s.p, 1)})`).join(" · ")}</div>
      </div>
    </div>
    <div class="card"><div class="explain" style="margin-top:0"><b>Dixon-Coles (1997)</b> models goals as two Poisson processes with
      <code>log λ_home = μ + home + attack_h − defence_a</code>. Parameters are fitted by <b>maximum likelihood</b> (L-BFGS-B) with exponential time decay
      <code>w = e^(−ξ·Δmatchday)</code> so recent form matters more, plus a correction factor ρ for the under-dispersed 0-0 / 1-0 / 0-1 / 1-1 scorelines.
      The xG variant fits the same structure on expected goals — a less noisy signal of quality. Elo converts rating difference into an expected score.</div></div>`;

  out.querySelector("#probs").innerHTML = Object.entries(p.models).map(([k, m]) => `
    <div><div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px">
      <span class="${k === "ensemble" ? "strong" : ""}" style="font-weight:${k === "ensemble" ? 650 : 500}">${names[k]}</span>
      <span class="muted mono">${fmt.pct(m.home)} · ${fmt.pct(m.draw)} · ${fmt.pct(m.away)}</span></div>
      <div style="display:flex;height:${k === "ensemble" ? 30 : 22}px;gap:2px;border-radius:7px;overflow:hidden">
        ${[[m.home, HOME(), p.home.short], [m.draw, css("--neutral"), "Draw"], [m.away, AWAY(), p.away.short]].map(([v, c, l]) =>
          `<div style="flex:${v};background:${c};color:#fff;font-size:11.5px;font-weight:650;display:grid;place-items:center">${v > 0.12 ? `${l} ${fmt.pct(v)}` : ""}</div>`).join("")}
      </div></div>`).join("");

  const M = p.score_matrix;
  const container = out.querySelector("#matrix");
  const size = Math.min(container.clientWidth, 460), n = M.length, pad = 26, cell = (size - pad) / n;
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${size} ${size}`).style("max-width", `${size}px`);
  const max = d3.max(M.flat());
  const scale = seqScale([0, max]);
  const cells = M.flatMap((row, i) => row.map((v, j) => ({ i, j, v })));
  const g = svg.append("g").attr("transform", `translate(${pad},${pad})`);
  g.selectAll("rect").data(cells).join("rect").attr("x", (c) => c.j * cell + 1).attr("y", (c) => c.i * cell + 1)
    .attr("width", cell - 2).attr("height", cell - 2).attr("rx", 4).attr("fill", (c) => scale(c.v))
    .attr("stroke", (c) => (c.i > c.j ? HOME() : c.i < c.j ? AWAY() : "none")).attr("stroke-opacity", 0.0)
    .on("mousemove", (e, c) => showTip(e, tipRows(`${p.home.short} ${c.i} – ${c.j} ${p.away.short}`, [["Probability", fmt.pct(c.v, 2)], ["Fair odds", (1 / c.v).toFixed(1)]])))
    .on("mouseleave", hideTip);
  const dark = document.documentElement.dataset.theme !== "light";
  g.selectAll("text.v").data(cells).join("text").attr("x", (c) => c.j * cell + cell / 2).attr("y", (c) => c.i * cell + cell / 2).attr("dy", "0.35em")
    .attr("text-anchor", "middle").attr("font-size", Math.min(12, cell / 3.6)).attr("pointer-events", "none")
    .attr("fill", (c) => ((c.v > max * 0.5) === dark ? "#0b0b0b" : (c.v > max * 0.5 ? "#fff" : css("--text-secondary"))))
    .text((c) => (c.v >= 0.005 ? (c.v * 100).toFixed(1) : ""));
  svg.selectAll("text.r").data(d3.range(n)).join("text").attr("x", pad - 8).attr("y", (i) => pad + i * cell + cell / 2).attr("dy", "0.35em").attr("text-anchor", "end").attr("font-size", 11).attr("fill", css("--text-muted")).text((i) => i);
  svg.selectAll("text.c").data(d3.range(n)).join("text").attr("y", pad - 8).attr("x", (i) => pad + i * cell + cell / 2).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", css("--text-muted")).text((i) => i);
}
