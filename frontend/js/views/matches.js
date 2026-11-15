/* global d3 */
import { api } from "../api.js";
import { AWAY, HOME, css, fmt, hideTip, legendHTML, lineChart, loading, pitch, responsive, showTip, tipRows } from "../charts.js";

export async function render(el, [matchId]) {
  const overview = await api("/api/overview");
  let md = overview.kpis.matchday;
  el.innerHTML = `
    <div class="grid" style="grid-template-columns:minmax(260px,320px) minmax(0,1fr)" id="mlayout">
      <div class="card" style="align-self:start">
        <div class="card-head"><h3 class="card-title">Fixtures</h3>
          <select class="select" id="md">${Array.from({ length: overview.kpis.matchday }, (_, i) => `<option value="${i + 1}">Matchday ${i + 1}</option>`).join("")}</select></div>
        <div class="match-list" id="mlist"></div>
      </div>
      <div class="grid" id="detail" style="align-content:start"></div>
    </div>`;
  if (window.innerWidth < 900) el.querySelector("#mlayout").style.gridTemplateColumns = "minmax(0,1fr)";
  const sel = el.querySelector("#md");
  const ros = [];
  const loadList = async (selectFirst) => {
    const list = (await api(`/api/matches?matchday=${md}`)).filter((m) => m.played);
    const box = el.querySelector("#mlist");
    box.innerHTML = list.map((m) => `<div class="match-item" data-id="${m.id}">
      <span class="h">${m.home}</span><span class="sc">${m.home_goals} – ${m.away_goals}</span><span class="a">${m.away}</span>
      <span class="xg">xG ${m.home_xg.toFixed(2)} – ${m.away_xg.toFixed(2)}</span></div>`).join("");
    box.querySelectorAll(".match-item").forEach((it) => it.addEventListener("click", () => { location.hash = `#/matches/${it.dataset.id}`; }));
    if (selectFirst && list.length) show(list[0].id);
  };
  const show = async (id) => {
    el.querySelectorAll(".match-item").forEach((it) => it.classList.toggle("active", +it.dataset.id === +id));
    const det = el.querySelector("#detail");
    loading(det, "Building pass networks…");
    const d = await api(`/api/matches/${id}`);
    ros.forEach((r) => r.disconnect()); ros.length = 0;
    detail(det, d, ros);
  };
  sel.addEventListener("change", () => { md = +sel.value; loadList(true); });
  if (matchId) {
    const m = await api(`/api/matches/${matchId}`).catch(() => null);
    if (m) md = m.match.matchday;
  }
  sel.value = md;
  await loadList(!matchId);
  if (matchId) show(matchId);
  return () => ros.forEach((r) => r.disconnect());
}

function detail(det, d, ros) {
  const h = d.home, a = d.away;
  det.innerHTML = `
    <div class="card">
      <div class="scoreboard">
        <div class="team"><span class="team-dot" style="background:${HOME()}"></span>${h.name}</div>
        <div><div class="score">${h.goals} – ${a.goals}</div><div class="xg-line">xG ${h.xg.toFixed(2)} – ${a.xg.toFixed(2)}</div></div>
        <div class="team away"><span class="team-dot" style="background:${AWAY()}"></span>${a.name}</div>
      </div>
      <div class="grid g-2" style="margin-top:18px">
        <div class="statbars" id="stats"></div>
        <div>
          <div class="card-sub" style="margin-bottom:8px">Deserved result — 10,000 replays of every shot as a Bernoulli trial with p = xG</div>
          <div id="deserved"></div>
        </div>
      </div>
    </div>
    <div class="grid g-2">
      <div class="card"><div class="card-head"><div><h3 class="card-title">Shot map</h3><div class="card-sub">Circle area ∝ xG · filled = goal · ${h.short} attack →, ${a.short} attack ←</div></div></div>
        <div class="chart" id="shotmap"></div>${legendHTML([{ label: h.name, color: HOME() }, { label: a.name, color: AWAY() }])}</div>
      <div class="card"><div class="card-head"><div><h3 class="card-title">xG race</h3><div class="card-sub">Cumulative expected goals · dots mark goals</div></div></div>
        <div class="chart" id="race"></div>${legendHTML([{ label: h.short, color: HOME() }, { label: a.short, color: AWAY() }], { line: true })}</div>
    </div>
    <div class="card"><div class="card-head"><div><h3 class="card-title">Momentum (xT generated per 5 minutes)</h3>
      <div class="card-sub">Positive threat created by successful passes &amp; carries — ${h.short} above the line, ${a.short} below</div></div></div>
      <div class="chart" id="momentum"></div></div>
    <div class="card">
      <div class="card-head"><div><h3 class="card-title">Pass network</h3>
        <div class="card-sub">Node = average touch position · size ∝ PageRank · edge width ∝ passes (≥3)</div></div>
        <div class="seg" id="team-seg"><button class="on" data-k="home">${h.short}</button><button data-k="away">${a.short}</button></div></div>
      <div class="grid g-21"><div class="chart" id="network"></div><div id="netstats"></div></div>
      <div class="explain"><b>Graph algorithms (hand-written, no networkx):</b> PageRank via power iteration on the pass matrix shows where possession <i>flows</i>;
        betweenness via Brandes' algorithm (Dijkstra with edge length 1/passes) shows who <i>bridges</i> units; degree centralisation measures over-reliance on one player.</div>
    </div>`;

  const bars = [["Possession", h.possession, a.possession, (v) => `${v}%`], ["Expected goals", h.xg, a.xg, (v) => v.toFixed(2)],
    ["Shots", h.shots, a.shots], ["On target", h.on_target, a.on_target], ["Pass accuracy", h.pass_pct, a.pass_pct, (v) => `${v}%`],
    ["Expected threat", h.xt, a.xt, (v) => v.toFixed(2)], ["Tackles + interceptions", h.tackles + h.interceptions, a.tackles + a.interceptions]];
  det.querySelector("#stats").innerHTML = bars.map(([l, hv, av, f = (v) => v]) => {
    const t = hv + av || 1;
    return `<div class="statbar"><div class="row"><b>${f(hv)}</b><span>${l}</span><b>${f(av)}</b></div>
      <div class="bars"><div style="flex:${hv / t};background:${HOME()}"></div><div style="flex:${av / t};background:${AWAY()}"></div></div></div>`;
  }).join("");
  const ds = d.deserved;
  det.querySelector("#deserved").innerHTML = `
    <div style="display:flex;height:34px;border-radius:8px;overflow:hidden;gap:2px">
      ${[[ds.home, HOME(), h.short], [ds.draw, css("--neutral"), "Draw"], [ds.away, AWAY(), a.short]].map(([p, c, l]) =>
        `<div style="flex:${p};background:${c};display:grid;place-items:center;color:#fff;font-size:12px;font-weight:650;min-width:${p > 0.08 ? 0 : 0}px">${p > 0.15 ? `${l} ${fmt.pct(p)}` : ""}</div>`).join("")}
    </div>
    <div class="kv" style="margin-top:14px">
      <span class="k">${h.short} win</span><span class="v">${fmt.pct(ds.home, 1)}</span>
      <span class="k">Draw</span><span class="v">${fmt.pct(ds.draw, 1)}</span>
      <span class="k">${a.short} win</span><span class="v">${fmt.pct(ds.away, 1)}</span>
      <span class="k">Network density (${h.short} / ${a.short})</span><span class="v">${fmt.pct(h.network.metrics.density)} / ${fmt.pct(a.network.metrics.density)}</span>
    </div>`;

  shotMap(det.querySelector("#shotmap"), d);
  const raceEl = det.querySelector("#race");
  const mk = (key) => [{ x: 0, y: 0 }, ...d.timeline.map((t) => ({ x: t.minute, y: t[key] })), { x: 95, y: d.timeline.length ? d.timeline[d.timeline.length - 1][key] : 0 }];
  ros.push(responsive(raceEl, () => lineChart(raceEl, {
    series: [{ name: h.short, color: HOME(), values: mk("home_cum") }, { name: a.short, color: AWAY(), values: mk("away_cum") }],
    curve: d3.curveStepAfter, height: 270, xLabel: "Minute", xDomain: [0, 95], yFormat: (v) => v.toFixed(1),
    markers: (svg, x, y) => {
      svg.selectAll(".goal").data(d.timeline.filter((t) => t.goal)).join("circle").attr("r", 6)
        .attr("cx", (t) => x(t.minute)).attr("cy", (t) => y(t.team_id === h.team_id ? t.home_cum : t.away_cum))
        .attr("fill", (t) => (t.team_id === h.team_id ? HOME() : AWAY())).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
    },
  })));
  const momEl = det.querySelector("#momentum");
  ros.push(responsive(momEl, () => momentum(momEl, d.momentum)));
  let side = "home";
  const netEl = det.querySelector("#network");
  const drawNet = () => network(netEl, det.querySelector("#netstats"), d[side], side === "home" ? HOME() : AWAY());
  det.querySelectorAll("#team-seg button").forEach((b) => b.addEventListener("click", () => {
    side = b.dataset.k; det.querySelectorAll("#team-seg button").forEach((x) => x.classList.toggle("on", x === b)); drawNet();
  }));
  drawNet();
}

function shotMap(container, d) {
  const { g } = pitch(container);
  const r = d3.scaleSqrt().domain([0, 1]).range([0.6, 4.2]);
  const homeId = d.home.team_id;
  const shots = d.shots.map((s) => ({ ...s, px: s.team_id === homeId ? s.x : 120 - s.x, py: s.team_id === homeId ? s.y : 80 - s.y }))
    .sort((a, b) => b.xg - a.xg);
  g.selectAll(".shot").data(shots).join("circle").attr("cx", (s) => s.px).attr("cy", (s) => s.py).attr("r", (s) => r(s.xg))
    .attr("fill", (s) => (s.outcome === "goal" ? (s.team_id === homeId ? HOME() : AWAY()) : "transparent"))
    .attr("fill-opacity", 0.9)
    .attr("stroke", (s) => (s.outcome === "goal" ? css("--surface-1") : s.team_id === homeId ? HOME() : AWAY()))
    .attr("stroke-width", (s) => (s.outcome === "goal" ? 0.4 : 0.45))
    .style("cursor", "pointer")
    .on("mousemove", (e, s) => showTip(e, tipRows(s.player, [["Minute", `${s.minute}'`], ["xG", s.xg.toFixed(3)], ["Outcome", s.outcome.replace("_", " ")],
      ["Situation", s.situation.replace("_", " ")], ["Body part", s.body_part]])))
    .on("mouseleave", hideTip);
}

function momentum(container, data) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const width = Math.max(container.clientWidth, 300), height = 200, m = { t: 10, r: 10, b: 26, l: 40 };
  const svg = el.append("svg").attr("viewBox", `0 0 ${width} ${height}`);
  const x = d3.scaleBand().domain(data.map((d) => d.minute)).range([m.l, width - m.r]).padding(0.18);
  const mx = d3.max(data, (d) => Math.max(d.home, d.away)) || 1;
  const y = d3.scaleLinear().domain([-mx, mx]).range([height - m.b, m.t]).nice();
  svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(width - m.l - m.r)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat((v) => Math.abs(v).toFixed(2)).tickSize(0).tickPadding(6)).select(".domain").remove();
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`).call(d3.axisBottom(x).tickValues(data.filter((d, i) => i % 3 === 0).map((d) => d.minute)).tickFormat((v) => `${v}'`).tickSizeOuter(0));
  const bw = x.bandwidth();
  svg.selectAll(".hb").data(data).join("path").attr("fill", HOME()).attr("d", (d) => roundedBar(x(d.minute), y(0) - 1, bw, y(0) - y(d.home), true));
  svg.selectAll(".ab").data(data).join("path").attr("fill", AWAY()).attr("d", (d) => roundedBar(x(d.minute), y(0) + 1, bw, y(-d.away) - y(0), false));
  svg.append("line").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--axis"));
  svg.selectAll(".hit").data(data).join("rect").attr("x", (d) => x(d.minute)).attr("width", bw).attr("y", m.t).attr("height", height - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", (e, d) => showTip(e, tipRows(`${d.minute}'–${d.minute + 5}'`, [["Home xT", d.home.toFixed(3)], ["Away xT", d.away.toFixed(3)]])))
    .on("mouseleave", hideTip);
}

// Bar anchored at the baseline with 4px rounding only on the data end.
function roundedBar(x, y0, w, h, up) {
  if (h <= 0.5) return "";
  const r = Math.min(4, w / 2, h);
  return up
    ? `M${x},${y0}V${y0 - h + r}Q${x},${y0 - h} ${x + r},${y0 - h}H${x + w - r}Q${x + w},${y0 - h} ${x + w},${y0 - h + r}V${y0}Z`
    : `M${x},${y0}V${y0 + h - r}Q${x},${y0 + h} ${x + r},${y0 + h}H${x + w - r}Q${x + w},${y0 + h} ${x + w},${y0 + h - r}V${y0}Z`;
}

function network(container, statsEl, side, color) {
  d3.select(container).selectAll("*").remove();
  const { g } = pitch(container);
  const net = side.network;
  const byId = new Map(net.nodes.map((n) => [n.id, n]));
  const maxC = d3.max(net.edges, (e) => e.count) || 1;
  const w = d3.scaleLinear().domain([3, maxC]).range([0.2, 1.6]);
  const r = d3.scaleSqrt().domain([0, d3.max(net.nodes, (n) => n.pagerank)]).range([0.8, 4.4]);
  g.selectAll(".edge").data(net.edges).join("line")
    .attr("x1", (e) => byId.get(e.source).x).attr("y1", (e) => byId.get(e.source).y)
    .attr("x2", (e) => byId.get(e.target).x).attr("y2", (e) => byId.get(e.target).y)
    .attr("stroke", color).attr("stroke-opacity", (e) => 0.25 + 0.6 * (e.count / maxC)).attr("stroke-width", (e) => w(e.count)).attr("stroke-linecap", "round")
    .on("mousemove", (ev, e) => showTip(ev, tipRows(`${byId.get(e.source).name} ⇄ ${byId.get(e.target).name}`, [["Total passes", e.count], ["→", e.a_to_b], ["←", e.b_to_a]])))
    .on("mouseleave", hideTip);
  const nodes = g.selectAll(".node").data(net.nodes).join("g").attr("transform", (n) => `translate(${n.x},${n.y})`).style("cursor", "pointer");
  nodes.append("circle").attr("r", (n) => r(n.pagerank)).attr("fill", color).attr("stroke", css("--surface-1")).attr("stroke-width", 0.5);
  nodes.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").attr("font-size", 2.2).attr("font-weight", 700).attr("fill", "#fff").text((n) => n.position);
  nodes.on("mousemove", (e, n) => showTip(e, tipRows(n.name, [["Position", n.position], ["PageRank", n.pagerank.toFixed(3)], ["Betweenness", n.betweenness.toFixed(3)],
    ["Passes made", n.passes_made], ["Passes received", n.passes_received]]))).on("mouseleave", hideTip);

  const top = [...net.nodes].sort((a, b) => b.pagerank - a.pagerank);
  const m = net.metrics;
  statsEl.innerHTML = `
    <div class="kv" style="margin-bottom:14px">
      <span class="k">Completed passes</span><span class="v">${m.total_passes}</span>
      <span class="k">Density</span><span class="v">${fmt.pct(m.density, 1)}</span>
      <span class="k">Degree centralisation</span><span class="v">${m.centralisation.toFixed(3)}</span>
      <span class="k">PageRank converged in</span><span class="v">${m.pagerank_iterations} iters</span>
    </div>
    <table class="data"><thead><tr><th>Player</th><th>PR</th><th>Betw.</th></tr></thead>
    <tbody>${top.slice(0, 7).map((n) => `<tr><td>${n.name} <span class="muted">${n.position}</span></td><td>${n.pagerank.toFixed(3)}</td><td>${n.betweenness.toFixed(3)}</td></tr>`).join("")}</tbody></table>`;
}
