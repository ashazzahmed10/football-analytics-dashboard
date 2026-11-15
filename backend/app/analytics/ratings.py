"""
Team-strength models and probabilistic forecasting.

* Elo            - online rating with goal-difference multiplier (World Football Elo style).
* Dixon-Coles    - bivariate Poisson goals model fitted by maximum likelihood with the
                   low-score dependence correction (rho) and exponential time decay (xi).
* xG-Poisson     - same structure, but fitted on expected goals instead of goals
                   (a de-noised signal of team quality).
* Monte Carlo    - vectorised simulation of the remaining fixtures -> finishing-position
                   probability matrix.
* Backtest       - walk-forward evaluation using the Ranked Probability Score (RPS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


# --------------------------------------------------------------------------- Elo
@dataclass
class Elo:
    k: float = 22.0
    home_adv: float = 60.0
    start: float = 1500.0

    def run(self, team_ids, matches):
        """matches: iterable of dicts with home_id, away_id, home_goals, away_goals, matchday."""
        r = {t: self.start for t in team_ids}
        history = {t: [{"matchday": 0, "elo": self.start}] for t in team_ids}
        for m in sorted(matches, key=lambda m: (m["matchday"], m["id"])):
            h, a = m["home_id"], m["away_id"]
            e_h = self.expected(r[h], r[a])
            gd = m["home_goals"] - m["away_goals"]
            s_h = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            g = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
            delta = self.k * g * (s_h - e_h)
            r[h] += delta
            r[a] -= delta
            for t in (h, a):
                history[t].append({"matchday": m["matchday"], "elo": round(r[t], 1)})
        self.ratings = r
        self.history = history
        return self

    def expected(self, rh, ra):
        return 1.0 / (1.0 + 10 ** (-(rh + self.home_adv - ra) / 400))

    def probs(self, h, a):
        e = self.expected(self.ratings[h], self.ratings[a])
        p_draw = max(0.12, 0.29 - 0.55 * abs(e - 0.5))
        p_home = max(0.01, e - p_draw / 2)
        p_away = max(0.01, 1 - e - p_draw / 2)
        s = p_home + p_draw + p_away
        return np.array([p_home, p_draw, p_away]) / s


# -------------------------------------------------------------------- Dixon-Coles
def dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles adjustment for 0-0, 1-0, 0-1, 1-1 (vectorised)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    """
    log(lambda_home) = mu + home + attack[h] - defence[a]
    log(lambda_away) = mu        + attack[a] - defence[h]
    Identifiability: sum(attack) = 0, sum(defence) = 0 (enforced by re-parameterisation).
    """

    def __init__(self, xi: float = 0.02, use_rho: bool = True):
        self.xi, self.use_rho = xi, use_rho

    def fit(self, team_ids, home_idx, away_idx, hg, ag, matchday):
        self.team_ids = list(team_ids)
        n = len(self.team_ids)
        self.n = n
        hg, ag = np.asarray(hg, float), np.asarray(ag, float)
        w = np.exp(-self.xi * (matchday.max() - matchday))
        integer = np.allclose(hg, np.round(hg)) and np.allclose(ag, np.round(ag))
        use_rho = self.use_rho and integer

        def unpack(p):
            att = np.append(p[:n - 1], -p[:n - 1].sum())
            dfn = np.append(p[n - 1:2 * n - 2], -p[n - 1:2 * n - 2].sum())
            mu, home = p[2 * n - 2], p[2 * n - 1]
            rho = p[2 * n] if use_rho else 0.0
            return att, dfn, mu, home, rho

        def nll(p):
            att, dfn, mu, home, rho = unpack(p)
            lam = np.exp(mu + home + att[home_idx] - dfn[away_idx])
            mu_ = np.exp(mu + att[away_idx] - dfn[home_idx])
            ll = hg * np.log(lam) - lam + ag * np.log(mu_) - mu_
            if use_rho:
                tau = dc_tau(hg, ag, lam, mu_, rho)
                ll = ll + np.log(np.clip(tau, 1e-10, None))
            return -np.sum(w * ll) + 0.01 * np.sum(p[:2 * n - 2] ** 2)

        p0 = np.zeros(2 * n + (1 if use_rho else 0))
        p0[2 * n - 2] = 0.2
        p0[2 * n - 1] = 0.25
        bounds = [(None, None)] * (2 * n) + ([(-0.2, 0.2)] if use_rho else [])
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        self.att, self.dfn, self.mu, self.home, self.rho = unpack(res.x)
        self.converged, self.nll, self.n_iter = bool(res.success), float(res.fun), int(res.nit)
        self.index = {t: i for i, t in enumerate(self.team_ids)}
        return self

    def rates(self, h, a):
        i, j = self.index[h], self.index[a]
        lam = math.exp(self.mu + self.home + self.att[i] - self.dfn[j])
        mu_ = math.exp(self.mu + self.att[j] - self.dfn[i])
        return lam, mu_

    def score_matrix(self, h, a):
        lam, mu_ = self.rates(h, a)
        g = np.arange(MAX_GOALS + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu_))
        if self.rho:
            M[0, 0] *= 1 - lam * mu_ * self.rho
            M[0, 1] *= 1 + lam * self.rho
            M[1, 0] *= 1 + mu_ * self.rho
            M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0.0, None)   # tau can go negative for extreme lambda*mu*rho
        return M / M.sum()

    def probs(self, h, a):
        M = self.score_matrix(h, a)
        return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])

    def ratings_table(self):
        return [{"team_id": t, "attack": float(self.att[i]), "defence": float(self.dfn[i]),
                 "overall": float(self.att[i] + self.dfn[i])} for t, i in self.index.items()]


def rps(probs: np.ndarray, outcome: int) -> float:
    """Ranked Probability Score for ordered outcomes (home, draw, away). Lower is better."""
    o = np.zeros(3)
    o[outcome] = 1
    return float(0.5 * np.sum((np.cumsum(probs)[:2] - np.cumsum(o)[:2]) ** 2))


# --------------------------------------------------------------------- Monte Carlo
def simulate_season(model: DixonColes, team_ids, played, remaining, n_sims=10000, seed=0):
    """
    played: list of (home, away, hg, ag); remaining: list of (home, away).
    Returns position-probability matrix and summary odds. Fully vectorised:
    every fixture's scoreline is drawn from its Dixon-Coles score matrix.
    """
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    pts0, gd0, gf0 = np.zeros(n), np.zeros(n), np.zeros(n)
    for h, a, hg, ag in played:
        i, j = idx[h], idx[a]
        gf0[i] += hg; gf0[j] += ag; gd0[i] += hg - ag; gd0[j] += ag - hg
        pts0[i] += 3 if hg > ag else 1 if hg == ag else 0
        pts0[j] += 3 if ag > hg else 1 if hg == ag else 0

    pts = np.tile(pts0, (n_sims, 1))
    gd = np.tile(gd0, (n_sims, 1))
    gf = np.tile(gf0, (n_sims, 1))
    size = MAX_GOALS + 1
    for h, a in remaining:
        M = model.score_matrix(h, a).ravel()
        draw = rng.choice(M.size, size=n_sims, p=M)
        hg, ag = draw // size, draw % size
        i, j = idx[h], idx[a]
        gf[:, i] += hg; gf[:, j] += ag
        gd[:, i] += hg - ag; gd[:, j] += ag - hg
        pts[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    key = pts * 1e6 + gd * 1e3 + gf + rng.random(pts.shape)   # tie-breakers: GD, GF, coin
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    position[rows, order] = np.arange(n)
    pos_matrix = np.stack([(position == k).mean(0) for k in range(n)], axis=1)  # team x position

    return [{
        "team_id": t,
        "current_points": int(pts0[idx[t]]),
        "expected_points": float(pts[:, idx[t]].mean()),
        "points_p10": float(np.percentile(pts[:, idx[t]], 10)),
        "points_p90": float(np.percentile(pts[:, idx[t]], 90)),
        "title": float(pos_matrix[idx[t], 0]),
        "top4": float(pos_matrix[idx[t], :4].sum()),
        "relegation": float(pos_matrix[idx[t], -3:].sum()),
        "expected_position": float((pos_matrix[idx[t]] * np.arange(1, n + 1)).sum()),
        "positions": pos_matrix[idx[t]].round(4).tolist(),
    } for t in team_ids]
