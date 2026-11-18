// SPA shell: hash router, lazy-loaded views, theme toggle, API status.
import { api } from "./api.js";

const ROUTES = {
  overview: { title: "League Overview", sub: "Standings, form and model-adjusted performance", mod: "overview" },
  matches: { title: "Match Centre", sub: "Shot maps, xG race, momentum and pass-network graph analysis", mod: "matches" },
  live: { title: "Live Match Replay", sub: "A real match streamed minute by minute over WebSocket, with a Bayesian in-play win-probability model", mod: "live" },
  xg: { title: "xG Lab", sub: "Logistic-regression expected-goals model trained from scratch (IRLS)", mod: "xg" },
  xt: { title: "Expected Threat", sub: "Markov-chain ball-progression model solved by value iteration", mod: "xt" },
  predict: { title: "Match Predictor", sub: "Dixon-Coles, xG-Poisson and Elo forecasts with an ensemble", mod: "predict" },
  season: { title: "Season Simulator", sub: "Vectorised Monte Carlo of every remaining fixture", mod: "season" },
  models: { title: "Model Validation", sub: "Walk-forward backtest and out-of-sample season tests", mod: "models" },
  scouting: { title: "Scouting Lab", sub: "PCA projection, k-means roles and cosine-similarity search", mod: "scouting" },
  architecture: { title: "Architecture & API", sub: "How the system is built", mod: "architecture" },
};

let cleanup = null;
const view = document.getElementById("view");

async function route() {
  const [, name = "overview", ...rest] = location.hash.split("/");
  const r = ROUTES[name] || ROUTES.overview;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (ROUTES[name] ? name : "overview")));
  document.getElementById("page-title").textContent = r.title;
  document.getElementById("page-sub").textContent = r.sub;
  document.title = `${r.title} · PitchIQ`;
  document.getElementById("sidebar").classList.remove("open");
  if (typeof cleanup === "function") cleanup();
  cleanup = null;
  view.innerHTML = "";
  window.scrollTo(0, 0);
  try {
    const mod = await import(`./views/${r.mod}.js`);
    cleanup = await mod.render(view, rest);
  } catch (err) {
    console.error(err);
    view.innerHTML = `<div class="card empty">Something went wrong: ${err.message}</div>`;
  }
}

function setupTheme() {
  document.getElementById("theme-btn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pitchiq-theme", next); } catch (e) { /* storage unavailable */ }
    route(); // re-render so charts pick up new tokens
  });
  document.getElementById("menu-btn").addEventListener("click", () => document.getElementById("sidebar").classList.toggle("open"));
}

async function status() {
  try {
    const [h, o] = await Promise.all([api("/api/health"), api("/api/overview")]);
    document.getElementById("api-status").textContent = `API online · ${(h.events / 1000).toFixed(0)}k events`;
    document.getElementById("md-chip").textContent = o.source ? o.source.competition : `Matchday ${o.kpis.matchday}`;
    document.getElementById("brand-sub").textContent = o.source ? o.source.competition : "Football analytics engine";
    const attr = document.getElementById("attribution");
    if (attr && o.source) attr.textContent = o.source.attribution;
  } catch (e) {
    document.getElementById("api-status").textContent = "API offline";
  }
}

window.addEventListener("hashchange", route);
setupTheme();
status();
route();
