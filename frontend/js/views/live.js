/* global d3 */
import { api, liveSocket } from "../api.js";
import { AWAY, HOME, css, fmt, hideTip, legendHTML, pitch, showTip, tipRows } from "../charts.js";

export async function render(el) {
  const all = (await api("/api/matches")).filter((m) => m.played);
  const mds = [...new Set(all.map((m) => m.matchday))].sort((a, b) => a - b);
  // default: the season's biggest xG shoot-out
  const pick = [...all].sort((a, b) => (b.home_xg + b.away_xg) - (a.home_xg + a.away_xg))[0];
  el.innerHTML = `
    <div class="card"><div class="controls" style="justify-content:space-between">
      <div class="controls">
        <select class="select" id="md">${mds.map((d) => `<option value="${d}" ${d === pick.matchday ? "selected" : ""}>Matchday ${d}</option>`).join("")}</select>
        <select class="select" id="match" style="max-width:320px"></select>
        <div class="seg" id="speed"><button data-v="2">2 min/s</button><button data-v="5" class="on">5 min/s</button><button data-v="15">15 min/s</button></div>
        <button class="btn primary" id="start">▶ Replay match</button>
      </div>
      <span class="chip" id="conn">Idle — WebSocket /ws/live</span>
    </div></div>
    <div class="card">
      <div class="scoreboard">
        <div class="team" id="hname"><span class="team-dot" style="background:${HOME()}"></span>Home</div>
        <div><div class="score" id="score">0 – 0</div><div class="xg-line" id="xgline">xG 0.00 – 0.00</div></div>
        <div class="team away" id="aname"><span class="team-dot" style="background:${AWAY()}"></span>Away</div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:14px"><span class="mono" id="minute" style="width:40px">0'</span><div class="progress" style="flex:1"><div id="prog" style="width:0%"></div></div></div>
    </div>
    <div class="grid g-21">
      <div class="card"><div class="card-head"><div><h3 class="card-title">In-play win probability</h3>
        <div class="card-sub">Bayesian update of each team's scoring rate from live xG, recomputed every minute of the replay</div></div></div>
        <div class="chart" id="wp"></div>
        <div style="margin-top:8px">${legendHTML([{ label: "Home win", color: HOME() }, { label: "Draw", color: css("--neutral") }, { label: "Away win", color: AWAY() }])}</div>
        <div class="explain">Prior: pre-match Dixon-Coles rates λ ~ Gamma(cλ, c), fitted only on <b>earlier</b> matchdays, so the model never sees the result it is forecasting. After fraction <i>t</i> of the match with live xG <i>x</i>,
          the posterior mean rate is <code>(cλ + x) / (c + t)</code>. Remaining goals ~ Poisson(rate · (1 − t)), convolved with the current score.
          <div class="kv" style="margin-top:10px" id="rates"></div></div>
      </div>
      <div class="grid" style="align-content:start">
        <div class="card"><h3 class="card-title" style="margin-bottom:12px">Live stats</h3><div class="statbars" id="lstats"></div></div>
        <div class="card"><h3 class="card-title" style="margin-bottom:10px">Event feed</h3><div class="feed" id="feed"><div class="muted">Waiting for kick-off…</div></div></div>
      </div>
    </div>
    <div class="card"><div class="card-head"><div><h3 class="card-title">Live pitch</h3><div class="card-sub">Shots sized by xG · arrows = high-threat passes/carries (ΔxT &gt; 0.03) · home attacks →</div></div></div>
      <div class="chart" id="lpitch" style="max-width:900px;margin:0 auto"></div></div>`;

  const fill = (selected) => {
    const md = +el.querySelector("#md").value;
    el.querySelector("#match").innerHTML = all.filter((m) => m.matchday === md).map((m) =>
      `<option value="${m.id}" ${m.id === selected ? "selected" : ""}>${m.home} ${m.home_goals}–${m.away_goals} ${m.away}</option>`).join("");
  };
  fill(pick.id);
  el.querySelector("#md").addEventListener("change", () => fill());
  let ws = null, speed = 5;
  el.querySelectorAll("#speed button").forEach((b) => b.addEventListener("click", () => {
    speed = +b.dataset.v; el.querySelectorAll("#speed button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  const startBtn = el.querySelector("#start");
  const stop = () => { if (ws) { ws.onclose = null; ws.close(); ws = null; } startBtn.textContent = "▶ Replay match"; };
  startBtn.addEventListener("click", () => {
    if (ws) { stop(); el.querySelector("#conn").className = "chip"; el.querySelector("#conn").textContent = "Stopped"; return; }
    start(el, +el.querySelector("#match").value, speed, (s) => { ws = s; }, stop);
    startBtn.textContent = "■ Stop";
  });
  return stop;
}

function start(el, matchId, speed, setWs, stop) {
  const ws = liveSocket({ match_id: matchId, speed });
  setWs(ws);
  const conn = el.querySelector("#conn");
  conn.className = "chip"; conn.textContent = "Connecting…";
  const probs = [];
  const feed = el.querySelector("#feed");
  const { g } = pitch(Object.assign(el.querySelector("#lpitch"), { innerHTML: "" }));
  const layer = g.append("g");
  let meta = null;

  ws.onmessage = (msg) => {
    const m = JSON.parse(msg.data);
    if (m.type === "error") { conn.textContent = m.detail; return; }
    if (m.type === "kickoff") {
      meta = m;
      conn.className = "chip live"; conn.textContent = "LIVE";
      el.querySelector("#hname").innerHTML = `<span class="team-dot" style="background:${HOME()}"></span>${m.home}`;
      el.querySelector("#aname").innerHTML = `<span class="team-dot" style="background:${AWAY()}"></span>${m.away}`;
      feed.innerHTML = "";
      probs.push({ minute: 0, ...m.probs });
      drawWP(el.querySelector("#wp"), probs);
      return;
    }
    if (m.type === "fulltime") {
      conn.className = "chip"; conn.textContent = `Full time · ${m.score[0]}–${m.score[1]} (xG ${m.xg[0]}–${m.xg[1]})`;
      const og = m.own_goals || [0, 0];
      if (og[0] || og[1]) {
        el.querySelector("#score").textContent = `${m.score[0]} – ${m.score[1]}`;
        addFeed(feed, "OG", `Own goal${og[0] + og[1] > 1 ? "s" : ""} added to the official score (own goals are not shots)`, "", false);
      }
      addFeed(feed, "FT", `Full time: ${meta.home} ${m.score[0]}–${m.score[1]} ${meta.away}`, "", true);
      stop();
      return;
    }
    // tick
    el.querySelector("#score").textContent = `${m.score[0]} – ${m.score[1]}`;
    el.querySelector("#xgline").textContent = `xG ${m.xg[0].toFixed(2)} – ${m.xg[1].toFixed(2)}`;
    el.querySelector("#minute").textContent = `${m.minute}'`;
    el.querySelector("#prog").style.width = `${Math.min(100, (m.minute / 94) * 100)}%`;
    probs.push({ minute: m.minute, ...m.probs });
    drawWP(el.querySelector("#wp"), probs);
    el.querySelector("#rates").innerHTML = `
      <span class="k">Prior source</span><span class="v" style="font-weight:500">${meta.prior}</span>
      <span class="k">Pre-match λ (home / away)</span><span class="v">${meta.lambda[0].toFixed(2)} / ${meta.lambda[1].toFixed(2)}</span>
      <span class="k">Posterior rate (home / away)</span><span class="v">${m.probs.rate_home.toFixed(2)} / ${m.probs.rate_away.toFixed(2)}</span>`;
    const stats = [["Possession", ...m.possession, (v) => `${v}%`], ["xG", ...m.xg, (v) => v.toFixed(2)], ["Shots", ...m.shots, (v) => v], ["xT created", ...m.xt, (v) => v.toFixed(2)]];
    el.querySelector("#lstats").innerHTML = stats.map(([l, hv, av, f]) => {
      const t = hv + av || 1;
      return `<div class="statbar"><div class="row"><b>${f(hv)}</b><span>${l}</span><b>${f(av)}</b></div>
        <div class="bars"><div style="flex:${hv / t || 0.5};background:${HOME()}"></div><div style="flex:${av / t || 0.5};background:${AWAY()}"></div></div></div>`;
    }).join("");
    for (const e of m.events) {
      const isHome = e.team_id === meta.home_id;
      const tx = (v) => (isHome ? v : 120 - v), ty = (v) => (isHome ? v : 80 - v);
      const color = isHome ? HOME() : AWAY();
      if (e.kind === "shot") {
        layer.append("circle").attr("cx", tx(e.x)).attr("cy", ty(e.y)).attr("r", 0).attr("fill", e.outcome === "goal" ? color : "transparent")
          .attr("stroke", e.outcome === "goal" ? css("--surface-1") : color).attr("stroke-width", 0.45)
          .on("mousemove", (ev) => showTip(ev, tipRows(e.player, [["Minute", `${m.minute}'`], ["xG", e.xg.toFixed(3)], ["Outcome", e.outcome.replace("_", " ")]])))
          .on("mouseleave", hideTip)
          .transition().duration(400).attr("r", d3.scaleSqrt().domain([0, 1]).range([0.6, 4.2])(e.xg));
        const label = e.outcome === "goal" ? `⚽ GOAL — ${e.player}` : `${e.player} shot (${e.outcome.replace("_", " ")})`;
        addFeed(feed, `${m.minute}'`, label, `xG ${e.xg.toFixed(2)}`, e.outcome === "goal", color);
      } else if (e.kind === "threat") {
        const a = layer.append("line").attr("x1", tx(e.x)).attr("y1", ty(e.y)).attr("x2", tx(e.end_x)).attr("y2", ty(e.end_y))
          .attr("stroke", color).attr("stroke-width", 0.5).attr("stroke-linecap", "round").attr("opacity", 0.9);
        a.transition().delay(2500).duration(1500).attr("opacity", 0).remove();
      }
    }
  };
  ws.onclose = () => { conn.className = "chip"; conn.textContent = "Disconnected"; stop(); };
  ws.onerror = () => { conn.textContent = "WebSocket error"; };
}

function addFeed(feed, min, text, right, goal, color) {
  const d = document.createElement("div");
  d.className = `feed-item ${goal ? "goal" : ""}`;
  d.innerHTML = `<span class="min">${min}</span><span>${color ? `<span class="team-dot" style="background:${color};margin-right:6px;width:8px;height:8px"></span>` : ""}${text}</span><span class="muted mono">${right}</span>`;
  feed.prepend(d);
}

function drawWP(container, data) {
  const el = d3.select(container);
  el.selectAll("*").remove();
  const W = Math.max(container.clientWidth, 300), H = 260, m = { t: 10, r: 12, b: 30, l: 40 };
  const svg = el.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain([0, 95]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  const stack = d3.stack().keys(["home", "draw", "away"])(data);
  const colors = [HOME(), css("--neutral"), AWAY()];
  const area = d3.area().x((d) => x(d.data.minute)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveMonotoneX);
  svg.selectAll("path").data(stack).join("path").attr("d", area).attr("fill", (d, i) => colors[i]).attr("stroke", css("--surface-1")).attr("stroke-width", 1.5);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([0, 15, 30, 45, 60, 75, 90]).tickFormat((v) => `${v}'`).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat((v) => fmt.pct(v)).tickSize(0).tickPadding(6)).select(".domain").remove();
  const last = data[data.length - 1];
  const cross = svg.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", css("--text-primary")).attr("opacity", 0);
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", (e) => {
      const mn = x.invert(d3.pointer(e)[0]);
      const d = data[d3.minIndex(data, (p) => Math.abs(p.minute - mn))];
      cross.attr("x1", x(d.minute)).attr("x2", x(d.minute)).attr("opacity", 0.5);
      showTip(e, tipRows(`${d.minute}'`, [["Home win", fmt.pct(d.home, 1)], ["Draw", fmt.pct(d.draw, 1)], ["Away win", fmt.pct(d.away, 1)]]));
    }).on("mouseleave", () => { cross.attr("opacity", 0); hideTip(); });
  if (last && last.minute >= 12) {
    const lx = x(last.minute) - 8;
    [["home", (last.home) / 2], ["draw", last.home + last.draw / 2], ["away", last.home + last.draw + last.away / 2]].forEach(([k, mid]) => {
      if (last[k] > 0.08) svg.append("text").attr("x", lx).attr("y", y(mid)).attr("dy", "0.35em").attr("text-anchor", "end").attr("font-size", 11.5).attr("font-weight", 700).attr("fill", "#fff").attr("pointer-events", "none").text(fmt.pct(last[k]));
    });
  }
}
