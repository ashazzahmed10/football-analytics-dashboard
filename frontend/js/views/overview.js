/* global d3 */
import { api } from "../api.js";
import { css, crest, fmt, hBars, hideTip, legendHTML, lineChart, loading, responsive, series, showTip, tipRows } from "../charts.js";

export async function render(el) {
  loading(el, "Loading league…");
  const [o, table, elo] = await Promise.all([api("/api/overview"), api("/api/league/table"), api("/api/league/elo")]);
  const k = o.kpis;
  el.innerHTML = `
    <div class="grid g-4">
      ${tile("Matches analysed", k.matches_played, `Matchday ${k.matchday} of ${k.total_matchdays}`)}
      ${tile("Events processed", fmt.int(k.events), `${fmt.int(k.shots)} shots · ${fmt.int(k.goals)} goals`)}
      ${tile("xG model ROC-AUC", k.xg_auc.toFixed(3), "Held-out 20% test set")}
      ${tile("Model pipeline", `${k.pipeline_seconds}s`, "Load → xG → xT → ratings → scouting")}
    </div>
    <div class="grid g-21">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">League table</h3>
          <div class="card-sub">xPts = expected points from re-simulating every match 2,000× from its shot-level xG</div></div></div>
        <div class="table-wrap"><table class="data" id="tbl"></table></div>
      </div>
      <div class="grid" style="align-content:start">
        <div class="card">
          <div class="card-head"><div><h3 class="card-title">Luck index</h3><div class="card-sub">Points − expected points</div></div></div>
          <div class="chart" id="luck"></div>
          <div class="card-note">Teams on the right are <b>out-performing</b> their chance quality — classic regression-to-the-mean candidates.</div>
        </div>
      </div>
    </div>
    <div class="grid g-21">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Elo rating race</h3><div class="card-sub">Rating after each matchday · top 4 highlighted</div></div><div id="elo-legend"></div></div>
        <div class="chart" id="elo"></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Top scorers: goals vs xG</h3><div class="card-sub">Finishing skill = gap between the two</div></div></div>
        <div class="chart" id="scorers"></div>
        <div style="margin-top:8px">${legendHTML([{ label: "Goals", color: css("--series-1") }, { label: "xG", color: css("--text-muted") }])}</div>
      </div>
    </div>`;

  drawTable(el.querySelector("#tbl"), table);
  const ros = [];
  const luckEl = el.querySelector("#luck");
  const luck = [...table].map((t) => ({ ...t, luck: t.pts - t.xpts })).sort((a, b) => b.luck - a.luck);
  ros.push(responsive(luckEl, () => hBars(luckEl, {
    data: luck, value: (d) => d.luck, label: (d) => d.short, labelW: 44, rowH: 19, diverging: true,
    color: (d) => (d.luck >= 0 ? css("--series-1") : css("--series-2")), format: (v) => fmt.signed(v, 1),
  })));

  const top = new Set(table.slice(0, 4).map((t) => t.team_id));
  const eloSeries = elo.map((t) => {
    const rank = table.findIndex((r) => r.team_id === t.team_id);
    return { name: t.name, label: t.short, color: top.has(t.team_id) ? series(rank) : css("--neutral"),
      muted: !top.has(t.team_id), width: top.has(t.team_id) ? 2.2 : 1.2, values: t.history.map((h) => ({ x: h.matchday, y: h.elo })) };
  });
  el.querySelector("#elo-legend").innerHTML = legendHTML(table.slice(0, 4).map((t, i) => ({ label: t.short, color: series(i) })), { line: true });
  const eloEl = el.querySelector("#elo");
  ros.push(responsive(eloEl, () => lineChart(eloEl, { series: eloSeries, height: 300, xLabel: "Matchday", yFormat: (v) => Math.round(v),
    curve: d3.curveLinear, yDomain: d3.extent(eloSeries.flatMap((s) => s.values), (v) => v.y).map((v, i) => v + (i ? 15 : -15)) })));

  const scEl = el.querySelector("#scorers");
  ros.push(responsive(scEl, () => dumbbell(scEl, o.top_scorers)));
  return () => ros.forEach((r) => r.disconnect());
}

const tile = (label, value, hint) => `<div class="card tile"><div class="label">${label}</div><div class="value">${value}</div><div class="hint">${hint}</div></div>`;

function drawTable(tbl, rows) {
  const cols = [
    ["#", "position"], ["Team", "name"], ["P", "p"], ["W", "w"], ["D", "d"], ["L", "l"], ["GD", "gd"],
    ["xGD", "xgd"], ["Pts", "pts"], ["xPts", "xpts"], ["Elo", "elo"], ["Form", null],
  ];
  let sortKey = "position", asc = true;
  const draw = () => {
    const data = [...rows].sort((a, b) => (asc ? 1 : -1) * (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0));
    tbl.innerHTML = `<thead><tr>${cols.map(([l, key]) => `<th class="${key ? "sortable" : ""} ${l === "Team" ? "left" : ""}" data-k="${key || ""}">${l}${sortKey === key ? (asc ? " ↑" : " ↓") : ""}</th>`).join("")}</tr></thead>
    <tbody>${data.map((r) => `<tr class="${r.position <= 4 ? "zone-ucl" : r.position >= 18 ? "zone-rel" : ""}">
      <td class="pos-num">${r.position}</td>
      <td class="left"><div class="team-cell">${crest(r.short)}<span>${r.name}</span></div></td>
      <td>${r.p}</td><td>${r.w}</td><td>${r.d}</td><td>${r.l}</td>
      <td class="${r.gd > 0 ? "delta-pos" : r.gd < 0 ? "delta-neg" : ""}">${fmt.signed(r.gd, 0)}</td>
      <td>${fmt.signed(r.xgd, 1)}</td><td class="strong">${r.pts}</td><td class="muted">${r.xpts.toFixed(1)}</td>
      <td>${r.elo}</td>
      <td><span class="form">${r.form.map((f) => `<span class="${f}">${f}</span>`).join("")}</span></td></tr>`).join("")}</tbody>`;
    tbl.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
      const key = th.dataset.k;
      if (sortKey === key) asc = !asc; else { sortKey = key; asc = key === "position" || key === "name"; }
      draw();
    }));
  };
  draw();
}

function dumbbell(container, data) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const width = Math.max(container.clientWidth, 280), rowH = 30, labelW = 120;
  const H = data.length * rowH + 26;
  const svg = el.append("svg").attr("viewBox", `0 0 ${width} ${H}`);
  const x = d3.scaleLinear().domain([0, d3.max(data, (d) => Math.max(d.goals, d.xg)) * 1.1]).range([labelW, width - 16]);
  svg.append("g").attr("class", "gridline").attr("transform", `translate(0,${H - 22})`)
    .call(d3.axisBottom(x).ticks(5).tickSize(-(H - 26)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 22})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  const g = svg.selectAll(".r").data(data).join("g").attr("transform", (d, i) => `translate(0,${i * rowH + 14})`);
  g.append("text").attr("x", labelW - 10).attr("dy", "0.32em").attr("text-anchor", "end").attr("font-size", 12)
    .attr("fill", css("--text-secondary")).text((d) => d.name);
  g.append("line").attr("x1", (d) => x(d.xg)).attr("x2", (d) => x(d.goals)).attr("stroke", css("--axis")).attr("stroke-width", 2);
  g.append("circle").attr("cx", (d) => x(d.xg)).attr("r", 5).attr("fill", css("--surface-1")).attr("stroke", css("--text-muted")).attr("stroke-width", 2);
  g.append("circle").attr("cx", (d) => x(d.goals)).attr("r", 5.5).attr("fill", css("--series-1")).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
  g.append("rect").attr("x", 0).attr("y", -rowH / 2).attr("width", width).attr("height", rowH).attr("fill", "transparent")
    .on("mousemove", (e, d) => showTip(e, tipRows(d.name, [["Team", d.team], ["Goals", d.goals], ["Assists", d.assists], ["xG", d.xg.toFixed(2)], ["Shots", d.shots], ["G − xG", fmt.signed(d.goals - d.xg)]])))
    .on("mouseleave", hideTip);
}
