"""
Live match centre: streams a simulated match over a WebSocket, minute by minute,
with an in-play win-probability model that updates in real time.

In-play model
-------------
Pre-match scoring rates (lambda_home, lambda_away) come from the Dixon-Coles model.
During the game we treat each team's true rate as uncertain and update it with a
Gamma-Poisson (conjugate) Bayesian update using the *xG it has generated so far*:

    prior      rate ~ Gamma(alpha = c * lambda, beta = c)          (c = prior strength in matches)
    posterior  E[rate] = (c * lambda + xG_so_far) / (c + t)        (t = fraction of match played)

Remaining goals ~ Poisson(E[rate] * (1 - t)); the final-result distribution is the
convolution of the current score with the remaining-goal distributions.
"""
from __future__ import annotations

import asyncio
import random
import sqlite3

import numpy as np
from scipy.stats import poisson

from .data.engine import MATCH_SECONDS, MatchSimulator
from .data.league import load_team_state

PRIOR_STRENGTH = 1.5


def in_play_probs(score_h, score_a, lam_h, lam_a, xg_h, xg_a, t):
    t = min(t, 0.999)
    rate_h = (PRIOR_STRENGTH * lam_h + xg_h) / (PRIOR_STRENGTH + t)
    rate_a = (PRIOR_STRENGTH * lam_a + xg_a) / (PRIOR_STRENGTH + t)
    rem = max(0.0, 1 - t)
    g = np.arange(11)
    ph, pa = poisson.pmf(g, rate_h * rem), poisson.pmf(g, rate_a * rem)
    M = np.outer(ph, pa)
    diff = (score_h - score_a) + g[:, None] - g[None, :]
    return {"home": float(M[diff > 0].sum()), "draw": float(M[diff == 0].sum()),
            "away": float(M[diff < 0].sum()), "rate_home": rate_h, "rate_away": rate_a}


async def stream_match(ws, platform, home: int, away: int, speed: float = 4.0, seed: int | None = None):
    """speed = simulated match-minutes per real second."""
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    con = sqlite3.connect(platform.db_path)
    hs, as_ = load_team_state(con, home, rng), load_team_state(con, away, rng)
    con.close()
    names = {p.id: p.name for p in hs.lineup + as_.lineup}
    lam_h, lam_a = platform.dc.rates(home, away)
    sim = MatchSimulator(hs, as_, rng)

    await ws.send_json({"type": "kickoff", "home": platform.team_name(home), "away": platform.team_name(away),
                        "home_id": home, "away_id": away, "lambda": [lam_h, lam_a],
                        "lineups": {"home": [{"id": p.id, "name": p.name, "pos": p.position} for p in hs.lineup],
                                    "away": [{"id": p.id, "name": p.name, "pos": p.position} for p in as_.lineup]},
                        "probs": in_play_probs(0, 0, lam_h, lam_a, 0, 0, 0)})

    state = {"xg": {home: 0.0, away: 0.0}, "passes": {home: 0, away: 0}, "shots": {home: 0, away: 0},
             "xt": {home: 0.0, away: 0.0}}
    minute, buffer = 0, []
    delay = 1.0 / max(speed, 0.1)

    async def flush(m):
        t = min(m * 60 / MATCH_SECONDS, 1.0)
        probs = in_play_probs(sim.score[home], sim.score[away], lam_h, lam_a,
                              state["xg"][home], state["xg"][away], t)
        tot = state["passes"][home] + state["passes"][away] or 1
        await ws.send_json({"type": "tick", "minute": m, "score": [sim.score[home], sim.score[away]],
                            "xg": [round(state["xg"][home], 3), round(state["xg"][away], 3)],
                            "xt": [round(state["xt"][home], 3), round(state["xt"][away], 3)],
                            "shots": [state["shots"][home], state["shots"][away]],
                            "possession": [round(state["passes"][home] / tot * 100, 1),
                                           round(state["passes"][away] / tot * 100, 1)],
                            "probs": probs, "events": buffer})
        await asyncio.sleep(delay)

    for ev in sim.run():
        m = ev.second // 60
        while m > minute:
            await flush(minute)
            buffer = []
            minute += 1
        if ev.type == "pass":
            state["passes"][ev.team_id] += 1
        if ev.type in ("pass", "carry") and ev.outcome == "success" and ev.end_x is not None:
            v = float(platform.xt.action_values([ev.x], [ev.y], [ev.end_x], [ev.end_y])[0])
            state["xt"][ev.team_id] += max(v, 0)
            if v > 0.03:
                buffer.append({"kind": "threat", "team_id": ev.team_id, "player": names[ev.player_id],
                               "x": ev.x, "y": ev.y, "end_x": ev.end_x, "end_y": ev.end_y, "xt": round(v, 3)})
        if ev.type == "shot":
            xg = float(platform.xg.predict([ev.x], [ev.y], [ev.body_part == "head"], [ev.situation])[0])
            state["xg"][ev.team_id] += xg
            state["shots"][ev.team_id] += 1
            buffer.append({"kind": "shot", "team_id": ev.team_id, "player": names[ev.player_id],
                           "x": ev.x, "y": ev.y, "xg": round(xg, 3), "outcome": ev.outcome,
                           "situation": ev.situation, "body_part": ev.body_part, "second": ev.second})
    await flush(minute)
    await ws.send_json({"type": "fulltime", "score": [sim.score[home], sim.score[away]],
                        "xg": [round(state["xg"][home], 2), round(state["xg"][away], 2)]})
