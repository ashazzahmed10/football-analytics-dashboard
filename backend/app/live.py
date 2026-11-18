"""
Live match centre: replays a real match over a WebSocket, minute by minute, with an
in-play win-probability model that updates as the events arrive.

In-play model
-------------
Pre-match scoring rates (lambda_home, lambda_away) come from a Dixon-Coles model fitted
ONLY on matchdays before this match (no look-ahead). During the game each team's true
rate is uncertain and is updated with a Gamma-Poisson (conjugate) Bayesian step using
the *xG it has generated so far*:

    prior      rate ~ Gamma(alpha = c * lambda, beta = c)          (c = prior strength in matches)
    posterior  E[rate] = (c * lambda + xG_so_far) / (c + t)        (t = fraction of match played)

Remaining goals ~ Poisson(E[rate] * (1 - t)); the final-result distribution is the
convolution of the current score with the remaining-goal distributions.
"""
from __future__ import annotations

import asyncio

import numpy as np
from scipy.stats import poisson

MATCH_MINUTES = 90
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


async def stream_replay(ws, platform, match_id: int, speed: float = 4.0):
    """speed = match-minutes per real second."""
    m = platform.matches.loc[match_id]
    home, away = int(m.home_id), int(m.away_id)
    ev = platform.events[platform.events.match_id == match_id].sort_values(["second", "id"])
    names = platform.players["name"].to_dict()
    lam_h, lam_a, prior_src = platform.prematch_rates(match_id)

    await ws.send_json({"type": "kickoff", "match_id": int(match_id), "matchday": int(m.matchday),
                        "date": m.get("date") if isinstance(m.get("date"), str) else None, "home": platform.team_name(home), "away": platform.team_name(away),
                        "home_id": home, "away_id": away, "lambda": [lam_h, lam_a], "prior": prior_src,
                        "final_score": [int(m.home_goals), int(m.away_goals)],
                        "probs": in_play_probs(0, 0, lam_h, lam_a, 0, 0, 0)})

    score = {home: 0, away: 0}
    xg = {home: 0.0, away: 0.0}
    xt = {home: 0.0, away: 0.0}
    passes = {home: 0, away: 0}
    shots = {home: 0, away: 0}
    delay = 1.0 / max(speed, 0.1)
    by_minute = ev.groupby(ev.second // 60)
    last_minute = int(ev.second.max() // 60) if len(ev) else 90

    for minute in range(0, last_minute + 1):
        feed = []
        if minute in by_minute.groups:
            for e in by_minute.get_group(minute).itertuples():
                tid = int(e.team_id)
                if tid not in score:
                    continue
                if e.type == "pass":
                    passes[tid] += 1
                if e.type in ("pass", "carry") and e.outcome == "success" and not np.isnan(e.xt):
                    xt[tid] += max(float(e.xt), 0)
                    if e.xt > 0.03:
                        feed.append({"kind": "threat", "team_id": tid, "player": names.get(e.player_id, ""),
                                     "x": e.x, "y": e.y, "end_x": e.end_x, "end_y": e.end_y, "xt": round(float(e.xt), 3)})
                if e.type == "shot":
                    xg[tid] += float(e.xg)
                    shots[tid] += 1
                    if e.outcome == "goal":
                        score[tid] += 1
                    feed.append({"kind": "shot", "team_id": tid, "player": names.get(e.player_id, ""),
                                 "x": e.x, "y": e.y, "xg": round(float(e.xg), 3), "outcome": e.outcome,
                                 "situation": e.situation, "body_part": e.body_part, "second": int(e.second)})
        t = min(minute / MATCH_MINUTES, 1.0)
        probs = in_play_probs(score[home], score[away], lam_h, lam_a, xg[home], xg[away], t)
        tot = passes[home] + passes[away] or 1
        await ws.send_json({"type": "tick", "minute": minute, "score": [score[home], score[away]],
                            "xg": [round(xg[home], 3), round(xg[away], 3)],
                            "xt": [round(xt[home], 3), round(xt[away], 3)],
                            "shots": [shots[home], shots[away]],
                            "possession": [round(passes[home] / tot * 100, 1), round(passes[away] / tot * 100, 1)],
                            "probs": probs, "events": feed})
        await asyncio.sleep(delay)

    # Own goals are not shots, so trust the official score at full time.
    await ws.send_json({"type": "fulltime", "score": [int(m.home_goals), int(m.away_goals)],
                        "xg": [round(xg[home], 2), round(xg[away], 2)],
                        "own_goals": [int(m.home_goals) - score[home], int(m.away_goals) - score[away]]})
