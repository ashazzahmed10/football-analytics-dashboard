"""
PitchIQ API - FastAPI application.

REST endpoints under /api, a WebSocket live-match stream under /ws, interactive
OpenAPI docs at /docs, and the single-page frontend served from /.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .live import stream_replay
from .services import Platform, resolve_database

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
# PITCHIQ_SOURCE=statsbomb (default, real Premier League 2015/16) or synthetic (simulated league).
# PITCHIQ_DB points at any prepared database and overrides both.
FRONTEND = ROOT / "frontend"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Path(os.environ["PITCHIQ_DB"]) if os.environ.get("PITCHIQ_DB") else \
        resolve_database(DATA_DIR, os.environ.get("PITCHIQ_SOURCE", "statsbomb"))
    STATE["platform"] = Platform(db)
    yield
    STATE.clear()


app = FastAPI(
    title="PitchIQ - Football Analytics Engine",
    version="1.0.0",
    description="Event-level football analytics: xG, xT, Dixon-Coles, Elo, Monte Carlo season "
                "simulation, pass-network graph analysis, PCA/k-means scouting and a live in-play model.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------- middleware
RATE_LIMIT, WINDOW = 120, 10.0           # 120 requests / 10 s per client IP (sliding window)
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def timing_and_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api"):
        ip = request.client.host if request.client else "anon"
        now = time.monotonic()
        q = _hits[ip]
        while q and now - q[0] > WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                headers={"Retry-After": str(int(WINDOW))})
        q.append(now)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def P() -> Platform:
    return STATE["platform"]


def _team_or_404(tid: int):
    if tid not in P().teams.index:
        raise HTTPException(404, f"team {tid} not found")


# ------------------------------------------------------------------- meta
@app.get("/api/health", tags=["meta"])
def health():
    p = P()
    return {"status": "ok", "events": int(len(p.events)), "pipeline_seconds": p.timings}


@app.get("/api/overview", tags=["league"])
def overview():
    return P().overview()


# ----------------------------------------------------------------- league
@app.get("/api/league/table", tags=["league"])
def league_table():
    return P().league_table()


@app.get("/api/league/elo", tags=["league"])
def elo_history():
    return P().elo_history()


@app.get("/api/league/ratings", tags=["models"], summary="Dixon-Coles team ratings + ground-truth recovery")
def ratings():
    return P().ratings()


@app.get("/api/teams", tags=["teams"])
def teams():
    t = P().teams
    return [{"id": int(r.id), "name": r.name, "short": r.short} for r in t.itertuples()]


@app.get("/api/teams/{team_id}", tags=["teams"])
def team(team_id: int):
    _team_or_404(team_id)
    return P().team_detail(team_id)


# ---------------------------------------------------------------- matches
@app.get("/api/matches", tags=["matches"])
def matches(matchday: Optional[int] = Query(None, ge=1, le=38), team_id: Optional[int] = None):
    return P().match_list(matchday, team_id)


@app.get("/api/matches/{match_id}", tags=["matches"], summary="Shots, xG timeline, pass networks, momentum")
def match(match_id: int):
    if match_id not in P().matches.index:
        raise HTTPException(404, "match not found")
    d = P().match_detail(match_id)
    if d is None:
        raise HTTPException(409, "match not played yet - try /api/predict")
    return d


# ----------------------------------------------------------------- models
class ShotIn(BaseModel):
    x: float = Field(..., ge=0, le=120, description="StatsBomb x (0-120, attacking right)")
    y: float = Field(..., ge=0, le=80)
    header: bool = False
    situation: Literal["open_play", "counter", "corner", "free_kick", "penalty"] = "open_play"


@app.post("/api/models/xg/predict", tags=["models"], summary="Score any shot with the trained xG model")
def xg_predict(shots: list[ShotIn]):
    if not shots or len(shots) > 1000:
        raise HTTPException(422, "send between 1 and 1000 shots")
    p = P().xg.predict([s.x for s in shots], [s.y for s in shots], [s.header for s in shots],
                       [s.situation for s in shots])
    return [{"xg": round(float(v), 4), **s.model_dump()} for v, s in zip(p, shots)]


@app.get("/api/models/xg", tags=["models"], summary="xG model report: metrics, coefficients, calibration, ROC")
def xg_report():
    return P().xg.report


@app.get("/api/models/xg/surface", tags=["models"])
def xg_surface(header: bool = False,
               situation: Literal["open_play", "counter", "corner", "free_kick"] = "open_play"):
    return P().xg.surface(header, situation)


@app.get("/api/models/xt", tags=["models"], summary="Expected-threat grid with value-iteration snapshots")
def xt_grid():
    return {**P().xt.to_dict(), "leaders": P().xt_leaders()}


@app.get("/api/predict", tags=["models"], summary="Pre-match prediction from 3 models + ensemble")
def predict(home: int, away: int):
    _team_or_404(home)
    _team_or_404(away)
    if home == away:
        raise HTTPException(422, "home and away must differ")
    return P().predict(home, away)


@app.get("/api/simulate/season", tags=["models"], summary="Monte Carlo simulation of the remaining fixtures")
def simulate(n: int = Query(10000, ge=100, le=50000),
             model: Literal["dixon_coles", "xg_poisson"] = "dixon_coles", seed: int = 0,
             as_of: Optional[int] = Query(None, ge=1, le=60,
                                          description="Fit on matchdays <= as_of, simulate the rest")):
    return P().season_simulation(n, model, seed, as_of)


@app.get("/api/simulate/title-race", tags=["models"], summary="Title/top-4/relegation odds after every matchday")
def title_race(model: Literal["dixon_coles", "xg_poisson"] = "xg_poisson"):
    return P().title_race(model)


@app.get("/api/models/halfway", tags=["models"], summary="Predict the second half of the season from the first")
def halfway(split: Optional[int] = Query(None, ge=4, le=60)):
    return P().halfway_test(split)


@app.get("/api/models/backtest", tags=["models"], summary="Walk-forward backtest (RPS / accuracy / log-loss)")
def backtest():
    return P().backtest()


# ---------------------------------------------------------------- players
@app.get("/api/players", tags=["players"])
def players(team_id: Optional[int] = None, position: Optional[str] = None,
            min_minutes: int = Query(0, ge=0), sort: str = "xt_p90", limit: int = Query(500, le=1000)):
    return P().player_list(team_id, position, min_minutes, sort, limit)


@app.get("/api/players/map", tags=["players"], summary="PCA scouting map + k-means roles")
def player_map():
    return P().scout.map()


@app.get("/api/players/{player_id}", tags=["players"])
def player(player_id: int):
    d = P().player_detail(player_id)
    if d is None:
        raise HTTPException(404, "player not found")
    return d


@app.get("/api/players/{player_id}/similar", tags=["players"])
def similar(player_id: int, n: int = Query(8, ge=1, le=30)):
    if player_id not in P().scout.index:
        raise HTTPException(404, "player not in scouting pool (needs 600+ minutes)")
    return P().scout.similar(player_id, n)


# ------------------------------------------------------------------- live
@app.websocket("/ws/live")
async def live(ws: WebSocket, match_id: int, speed: float = 4.0):
    """Replay a played match minute by minute with the in-play win-probability model."""
    await ws.accept()
    p = P()
    if match_id not in p.matches.index or not p.matches.loc[match_id, "played"]:
        await ws.send_json({"type": "error", "detail": "unknown or unplayed match"})
        await ws.close()
        return
    try:
        await stream_replay(ws, p, match_id, min(max(speed, 0.5), 60.0))
        await ws.close()
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------- frontend
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND / "index.html")


# A service worker left behind by another app on the same localhost port keeps serving
# that app's cached pages. Browsers re-fetch /sw.js to update it, so we answer with a
# "kill switch" worker that wipes its caches, unregisters itself and reloads the tabs.
KILL_SWITCH_SW = """
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) await caches.delete(key);
    await self.registration.unregister();
    for (const client of await self.clients.matchAll({ type: 'window' })) client.navigate(client.url);
  })());
});
"""


@app.get("/sw.js", include_in_schema=False)
@app.get("/service-worker.js", include_in_schema=False)
def kill_stale_service_worker():
    return Response(KILL_SWITCH_SW, media_type="application/javascript",
                    headers={"Cache-Control": "no-store", "Clear-Site-Data": '"cache", "storage"'})
