// Shared chart primitives built on D3 v7. Every colour is read from CSS tokens so
// charts follow the light/dark theme.
/* global d3 */

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const series = (i) => css(`--series-${(i % 8) + 1}`);
export const HOME = () => css("--series-1");
export const AWAY = () => css("--series-2");

export const fmt = {
  pct: (v, d = 0) => `${(v * 100).toFixed(d)}%`,
  n: (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d)),
  int: (v) => d3.format(",")(Math.round(v)),
  signed: (v, d = 2) => `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}`,
};

export function isDark() {
  return document.documentElement.dataset.theme !== "light";
}

// Sequential ramp (one hue, near-surface -> saturated) and diverging ramp (blue <-> gray <-> red)
export function seqScale(domain = [0, 1]) {
  const stops = isDark()
    ? ["#1c2533", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]
    : ["#eef4fd", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"];
  return d3.scaleSequential(d3.interpolateRgbBasis(stops)).domain(domain);
}
export function divScale(max = 1) {
  const mid = css("--div-mid");
  const stops = isDark() ? ["#3987e5", mid, "#e66767"] : ["#2a78d6", mid, "#e34948"];
  return d3.scaleDiverging(d3.interpolateRgbBasis(stops)).domain([-max, 0, max]);
}

// ------------------------------------------------------------------ tooltip
const tt = () => document.getElementById("tooltip");
export function showTip(event, html) {
  const el = tt();
  el.innerHTML = html;
  el.style.opacity = 1;
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const r = el.getBoundingClientRect();
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + r.width > W - 8) x = event.clientX - r.width - pad;
  if (y + r.height > H - 8) y = event.clientY - r.height - pad;
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}
export function hideTip() { tt().style.opacity = 0; }
export const tipRows = (title, rows) =>
  `<div class="tt-title">${title}</div>` + rows.map(([k, v]) => `<div class="tt-row"><span>${k}</span><b>${v}</b></div>`).join("");

// -------------------------------------------------------------------- pitch
// Draws a pitch in StatsBomb units (120 x 80). Returns the <g> to draw into.
export function pitch(container, { half = false, pad = 2, vertical = false } = {}) {
  const x0 = half ? 60 : 0;
  const w = 120 - x0;
  const svg = d3.select(container).append("svg")
    .attr("viewBox", `${x0 - pad} ${-pad} ${w + 2 * pad} ${80 + 2 * pad}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g");
  g.append("rect").attr("class", "pitch-bg").attr("x", x0).attr("y", 0).attr("width", w).attr("height", 80).attr("rx", 0.6);
  const L = g.append("g").attr("class", "pitch-markings");
  const line = (attrs, tag = "rect") => { const e = L.append(tag).attr("class", "pitch-line"); Object.entries(attrs).forEach(([k, v]) => e.attr(k, v)); return e; };
  line({ x: x0, y: 0, width: w, height: 80 });
  if (!half) line({ x1: 60, y1: 0, x2: 60, y2: 80 }, "line");
  if (half) L.append("path").attr("class", "pitch-line").attr("d", d3.arc()({ innerRadius: 10, outerRadius: 10, startAngle: 0, endAngle: Math.PI })).attr("transform", "translate(60,40)");
  else line({ cx: 60, cy: 40, r: 10 }, "circle");
  for (const side of half ? [1] : [0, 1]) {
    const gx = side ? 120 : 0, dir = side ? -1 : 1;
    line({ x: side ? 102 : 0, y: 18, width: 18, height: 44 });
    line({ x: side ? 114 : 0, y: 30, width: 6, height: 20 });
    line({ cx: gx + dir * 12, cy: 40, r: 0.4 }, "circle").style("fill", css("--pitch-line"));
    // penalty "D": the part of the 10-yd circle around the spot that lies outside the box
    const c = side ? -Math.PI / 2 : Math.PI / 2, half = Math.acos(0.6);
    const arc = d3.arc()({ innerRadius: 10, outerRadius: 10, startAngle: c - half, endAngle: c + half });
    L.append("path").attr("class", "pitch-line").attr("d", arc).attr("transform", `translate(${gx + dir * 12},40)`);
    line({ x: side ? 120 : -2, y: 36, width: 2, height: 8 });
  }
  if (half) {
    // mask the centre-circle overflow on half pitch
    svg.attr("style", "overflow:hidden");
  }
  return { svg, g };
}

// --------------------------------------------------------------- line chart
// series: [{ name, color, values: [{x, y}], width, dash, muted }]
export function lineChart(container, {
  series, height = 260, xLabel = "", yLabel = "", xDomain, yDomain, curve = d3.curveMonotoneX,
  yFormat = (v) => v, xFormat = (v) => v, margin = { top: 12, right: 16, bottom: 34, left: 44 }, area = false,
  markers = null, endLabels = false,
}) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const width = Math.max(container.clientWidth, 300);
  const svg = el.append("svg").attr("viewBox", `0 0 ${width} ${height}`);
  const all = series.flatMap((s) => s.values);
  const x = d3.scaleLinear().domain(xDomain || d3.extent(all, (d) => d.x)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain || [Math.min(0, d3.min(all, (d) => d.y)), d3.max(all, (d) => d.y) * 1.08])
    .nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class", "gridline").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(Math.min(10, width / 70)).tickFormat(xFormat).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(yFormat).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
  if (xLabel) svg.append("text").attr("class", "axis-title").attr("x", width - margin.right).attr("y", height - 4).attr("text-anchor", "end").text(xLabel);
  if (yLabel) svg.append("text").attr("class", "axis-title").attr("x", margin.left - 36).attr("y", margin.top - 2).text(yLabel);

  const gen = d3.line().x((d) => x(d.x)).y((d) => y(d.y)).curve(curve);
  const ordered = [...series].sort((a, b) => (a.muted ? 0 : 1) - (b.muted ? 0 : 1));
  if (area) {
    const ag = d3.area().x((d) => x(d.x)).y0(y(0)).y1((d) => y(d.y)).curve(curve);
    ordered.forEach((s) => svg.append("path").attr("d", ag(s.values)).attr("fill", s.color).attr("opacity", 0.12));
  }
  ordered.forEach((s) => {
    svg.append("path").attr("d", gen(s.values)).attr("fill", "none").attr("stroke", s.color)
      .attr("stroke-width", s.width || 2).attr("stroke-dasharray", s.dash || null)
      .attr("opacity", s.muted ? 0.35 : 1).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
    if (endLabels && !s.muted) {
      const last = s.values[s.values.length - 1];
      svg.append("text").attr("x", x(last.x) + 6).attr("y", y(last.y) + 4).attr("font-size", 11)
        .attr("fill", css("--text-secondary")).attr("font-weight", 600).text(s.label || s.name);
    }
  });
  if (markers) markers(svg, x, y);

  // crosshair + tooltip
  const xs = [...new Set(all.map((d) => d.x))].sort((a, b) => a - b);
  const cross = svg.append("line").attr("stroke", css("--axis")).attr("y1", margin.top).attr("y2", height - margin.bottom).style("opacity", 0);
  const dots = svg.append("g");
  svg.append("rect").attr("x", margin.left).attr("y", margin.top).attr("width", width - margin.left - margin.right)
    .attr("height", height - margin.top - margin.bottom).attr("fill", "transparent")
    .on("mousemove", (event) => {
      const [mx] = d3.pointer(event);
      const xv = x.invert(mx);
      const xi = xs[d3.minIndex(xs, (v) => Math.abs(v - xv))];
      cross.attr("x1", x(xi)).attr("x2", x(xi)).style("opacity", 1);
      const rows = series.filter((s) => !s.muted || series.length < 6).map((s) => {
        const pt = s.values.reduce((best, d) => (d.x <= xi && (!best || d.x > best.x) ? d : best), null) || s.values[0];
        return { s, pt };
      }).sort((a, b) => b.pt.y - a.pt.y);
      dots.selectAll("circle").data(rows).join("circle").attr("r", 4).attr("cx", x(xi)).attr("cy", (d) => y(d.pt.y))
        .attr("fill", (d) => d.s.color).attr("stroke", css("--surface-1")).attr("stroke-width", 2);
      showTip(event, tipRows(`${xLabel ? xLabel + " " : ""}${xFormat(xi)}`, rows.slice(0, 8).map((r) => [r.s.name, yFormat(r.pt.y)])));
    })
    .on("mouseleave", () => { cross.style("opacity", 0); dots.selectAll("*").remove(); hideTip(); });
  return { svg, x, y };
}

// --------------------------------------------------------- horizontal bars
export function hBars(container, { data, value, label, color, height, format = (v) => v, max, rowH = 24, labelW = 130, diverging = false }) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const width = Math.max(container.clientWidth, 280);
  const H = height || data.length * rowH + 10;
  const svg = el.append("svg").attr("viewBox", `0 0 ${width} ${H}`);
  const ext = max ?? d3.max(data, (d) => Math.abs(value(d)));
  const x = diverging
    ? d3.scaleLinear().domain([-ext, ext]).range([labelW + 42, width - 50])
    : d3.scaleLinear().domain([0, ext]).range([labelW, width - 50]);
  const zero = x(0);
  const g = svg.selectAll("g").data(data).join("g").attr("transform", (d, i) => `translate(0,${i * rowH + 4})`);
  g.append("text").attr("x", labelW - 8).attr("y", rowH / 2).attr("dy", "0.32em").attr("text-anchor", "end")
    .attr("font-size", 12).attr("fill", css("--text-secondary")).text(label);
  g.append("rect").attr("x", (d) => Math.min(zero, x(value(d)))).attr("y", 4).attr("height", rowH - 10)
    .attr("width", (d) => Math.max(1, Math.abs(x(value(d)) - zero))).attr("rx", 4)
    .attr("fill", (d, i) => (typeof color === "function" ? color(d, i) : color));
  g.append("text").attr("x", (d) => (value(d) >= 0 ? Math.max(zero, x(value(d))) + 6 : Math.min(zero, x(value(d))) - 6))
    .attr("text-anchor", (d) => (value(d) >= 0 ? "start" : "end"))
    .attr("y", rowH / 2).attr("dy", "0.32em").attr("font-size", 11.5).attr("fill", css("--text-primary"))
    .attr("font-variant-numeric", "tabular-nums").text((d) => format(value(d)));
  if (diverging) svg.append("line").attr("x1", zero).attr("x2", zero).attr("y1", 0).attr("y2", H).attr("stroke", css("--axis"));
  g.append("rect").attr("x", 0).attr("width", width).attr("height", rowH).attr("fill", "transparent")
    .on("mousemove", (e, d) => showTip(e, tipRows(label(d), [["Value", format(value(d))]])))
    .on("mouseleave", hideTip);
  return svg;
}

// ------------------------------------------------------------ pitch heatmap
export function pitchGrid(g, grid, { x0 = 0, scale, opacity = 0.92, onHover } = {}) {
  const ny = grid.length, nx = grid[0].length;
  const cw = (120 - x0) / nx, ch = 80 / ny;
  const flat = grid.flatMap((row, j) => row.map((v, i) => ({ i, j, v })));
  const s = scale || seqScale([0, d3.max(flat, (d) => d.v)]);
  return g.insert("g", ".pitch-markings").selectAll("rect").data(flat).join("rect")
    .attr("x", (d) => x0 + d.i * cw).attr("y", (d) => d.j * ch).attr("width", cw + 0.02).attr("height", ch + 0.02)
    .attr("fill", (d) => s(d.v)).attr("opacity", opacity)
    .on("mousemove", (e, d) => onHover && onHover(e, d))
    .on("mouseleave", hideTip);
}

// ------------------------------------------------------------- legend html
export function legendHTML(items, { line = false } = {}) {
  return `<div class="legend">${items.map((it) => `<span class="key"><span class="sw ${line ? "line" : ""}" style="background:${it.color}"></span>${it.label}</span>`).join("")}</div>`;
}

export function crest(short, color) {
  return `<span class="crest" style="${color ? `background:${color};color:#fff` : ""}">${short}</span>`;
}

export function loading(el, msg = "Loading…") {
  el.innerHTML = `<div class="loader"><div class="spinner"></div><div>${msg}</div></div>`;
}

// Keep charts sharp on resize: re-run a draw fn when its container width changes.
export function responsive(el, draw) {
  let w = el.clientWidth;
  const ro = new ResizeObserver(() => {
    if (Math.abs(el.clientWidth - w) > 20) { w = el.clientWidth; draw(); }
  });
  ro.observe(el);
  draw();
  return ro;
}
