/* global d3 */
import { api } from "../api.js";
import { css, fmt, hideTip, loading, pitch, pitchGrid, seqScale, series, showTip, tipRows } from "../charts.js";

const SHAPES = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolStar, d3.symbolWye, d3.symbolCross, d3.symbolCircle];

export async function render(el, [pid]) {
  loading(el, "Fitting PCA + k-means…");
  const map = await api("/api/players/map");
  const hidden = new Set();
  el.innerHTML = `
    <div class="grid g-21">
      <div class="card">
        <div class="card-head"><div><h3 class="card-title">Scouting map</h3>
          <div class="card-sub">${map.players.length} players (600+ min) · 14 per-90 features → 2 principal components (${fmt.pct(map.explained_variance[0] + map.explained_variance[1])} of variance)</div></div>
          <div class="controls"><input class="input" id="search" list="plist" placeholder="Search player…" style="width:170px" />
            <datalist id="plist">${map.players.map((p) => `<option value="${p.name}">`).join("")}</datalist>
            <label class="chip" style="cursor:pointer"><input type="checkbox" id="biplot" style="margin:0" /> Loadings</label></div></div>
        <div class="chart" id="scatter"></div>
        <div class="legend" id="clegend" style="margin-top:10px"></div>
        <div class="explain"><b>Unsupervised role discovery.</b> Features are z-scored, projected with <b>PCA (SVD)</b>, and clustered with <b>k-means++</b>.
          k is chosen automatically by <b>silhouette score</b> (k = ${map.k}: ${map.k_selection.map((s) => `k=${s.k} → ${s.silhouette.toFixed(3)}`).join(", ")}).
          Role names are generated from each centroid's most distinctive trait relative to its positional group. Click a legend item to hide a cluster.</div>
      </div>
      <div class="card" id="profile" style="align-self:start"><div class="empty">Click a player on the map</div></div>
    </div>`;

  const legend = el.querySelector("#clegend");
  const drawLegend = () => {
    legend.innerHTML = map.clusters.map((c) => `<span class="key toggle ${hidden.has(c.id) ? "off" : ""}" data-id="${c.id}">
      <svg width="12" height="12" viewBox="-6 -6 12 12"><path d="${d3.symbol(SHAPES[c.id], 60)()}" fill="${series(c.id)}"/></svg>${c.name} (${c.size})</span>`).join("");
    legend.querySelectorAll(".key").forEach((k) => k.addEventListener("click", () => {
      const id = +k.dataset.id; hidden.has(id) ? hidden.delete(id) : hidden.add(id); drawLegend(); draw();
    }));
  };
  let selected = pid ? +pid : null;
  const scatterEl = el.querySelector("#scatter");
  const draw = () => scatter(scatterEl, map, hidden, selected, el.querySelector("#biplot").checked, (id) => { selected = id; draw(); profile(el.querySelector("#profile"), id, select); });
  const select = (id) => { selected = id; draw(); profile(el.querySelector("#profile"), id, select); };
  el.querySelector("#biplot").addEventListener("change", draw);
  el.querySelector("#search").addEventListener("change", (e) => {
    const p = map.players.find((x) => x.name.toLowerCase() === e.target.value.toLowerCase());
    if (p) select(p.player_id);
  });
  drawLegend();
  draw();
  const ro = new ResizeObserver(() => draw());
  ro.observe(scatterEl);
  select(selected || [...map.players].sort((a, b) => b.pc1 - a.pc1)[0].player_id);
  return () => ro.disconnect();
}

function scatter(container, map, hidden, selected, biplot, onClick) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const W = Math.max(container.clientWidth, 300), H = Math.min(520, W * 0.75), m = { t: 10, r: 14, b: 34, l: 40 };
  const svg = el.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const P = map.players;
  const x = d3.scaleLinear().domain(d3.extent(P, (p) => p.pc1)).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(P, (p) => p.pc2)).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(W - m.l - m.r)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(0).tickPadding(6)).select(".domain").remove();
  svg.append("text").attr("class", "axis-title").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").text(`PC1 (${fmt.pct(map.explained_variance[0])})`);
  svg.append("text").attr("class", "axis-title").attr("x", m.l + 4).attr("y", m.t + 10).text(`PC2 (${fmt.pct(map.explained_variance[1])})`);
  if (biplot) {
    const sc = Math.min(x.domain()[1], y.domain()[1]) * 0.9 / d3.max(map.loadings, (l) => Math.hypot(l.pc1, l.pc2));
    const g = svg.append("g").attr("pointer-events", "none");
    map.loadings.forEach((l) => {
      g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(l.pc1 * sc)).attr("y2", y(l.pc2 * sc)).attr("stroke", css("--text-muted")).attr("stroke-width", 1.2);
      g.append("text").attr("x", x(l.pc1 * sc * 1.08)).attr("y", y(l.pc2 * sc * 1.08)).attr("font-size", 10.5).attr("text-anchor", "middle").attr("fill", css("--text-secondary")).text(l.feature);
    });
  }
  const vis = P.filter((p) => !hidden.has(p.cluster));
  svg.append("g").selectAll("path").data(vis).join("path")
    .attr("d", (p) => d3.symbol(SHAPES[p.cluster], p.player_id === selected ? 180 : 52)())
    .attr("transform", (p) => `translate(${x(p.pc1)},${y(p.pc2)})`)
    .attr("fill", (p) => series(p.cluster)).attr("fill-opacity", (p) => (selected && p.player_id !== selected ? 0.75 : 1))
    .attr("stroke", (p) => (p.player_id === selected ? css("--text-primary") : css("--surface-1"))).attr("stroke-width", (p) => (p.player_id === selected ? 2 : 1))
    .style("cursor", "pointer")
    .on("mousemove", (e, p) => showTip(e, tipRows(p.name, [["Team", p.team], ["Position", p.position], ["Role", p.role], ["Minutes", p.minutes]])))
    .on("mouseleave", hideTip).on("click", (e, p) => onClick(p.player_id));
  const sp = P.find((p) => p.player_id === selected);
  if (sp && !hidden.has(sp.cluster)) {
    svg.append("text").attr("x", x(sp.pc1)).attr("y", y(sp.pc2) - 12).attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 650)
      .attr("fill", css("--text-primary")).attr("paint-order", "stroke").attr("stroke", css("--surface-1")).attr("stroke-width", 3).text(sp.name);
  }
}

async function profile(box, id, select) {
  loading(box);
  const d = await api(`/api/players/${id}`);
  const p = d.player, pr = d.profile;
  box.innerHTML = `
    <div class="card-head"><div><h3 class="card-title" style="font-size:18px">${p.name}</h3>
      <div class="card-sub">${p.team} · ${p.position} · age ${p.age} · ${fmt.int(p.minutes)} min</div></div>
      ${pr ? `<span class="chip"><svg width="10" height="10" viewBox="-6 -6 12 12"><path d="${d3.symbol(SHAPES[pr.cluster], 60)()}" fill="${series(pr.cluster)}"/></svg>${pr.role}</span>` : ""}</div>
    <div class="grid g-4" style="gap:8px;margin-bottom:14px">
      ${[["Goals", p.goals], ["Assists", p.assists], ["xG", p.xg.toFixed(1)], ["xT", p.xt.toFixed(2)]].map(([k, v]) => `<div style="background:var(--surface-2);border-radius:10px;padding:8px 10px"><div class="muted" style="font-size:11px">${k}</div><div style="font-size:18px;font-weight:700">${v}</div></div>`).join("")}
    </div>
    ${pr ? `<div class="card-sub" style="margin-bottom:6px">Percentile vs. all scouted players</div>
      ${pr.percentiles.map((q) => `<div class="pct-row"><span>${q.label}</span><div class="track"><div class="fill" style="width:${q.percentile}%;background:${q.percentile >= 80 ? css("--series-1") : q.percentile >= 40 ? css("--series-3") : css("--neutral")}"></div></div>
        <span class="num">${Math.round(q.percentile)}</span></div>`).join("")}` : ""}
    <div class="card-sub" style="margin:14px 0 6px">Touch heatmap (passes &amp; carries)</div>
    <div class="chart" id="pheat"></div>
    ${d.similar ? `<div class="card-sub" style="margin:14px 0 6px">Most similar players (cosine similarity, z-scored features)</div>
      <div class="sim-list">${d.similar.map((s) => `<div class="sim-item" data-id="${s.player_id}"><div>${s.name}<div class="meta">${s.team} · ${s.position} · ${s.role}</div></div>
        <div style="text-align:right;font-variant-numeric:tabular-nums;font-weight:650">${(s.similarity * 100).toFixed(1)}%</div></div>`).join("")}</div>` : ""}`;
  const { g } = pitch(box.querySelector("#pheat"));
  pitchGrid(g, d.heatmap, { scale: seqScale([0, 1]), opacity: 0.95 });
  box.querySelectorAll(".sim-item").forEach((it) => it.addEventListener("click", () => select(+it.dataset.id)));
}
