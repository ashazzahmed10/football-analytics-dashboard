"""
Analytics platform: loads the event store, trains every model once at start-up and
serves cached, JSON-ready results to the API layer.

Pipeline (runs in ~3-5 s):
    SQLite events -> xG model (IRLS) -> xT (value iteration) -> per-action values
                  -> Elo / Dixon-Coles (goals & xG) -> player per-90 features
                  -> PCA + k-means scouting model
"""
from __future__ import annotations

from typing import Optional

import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics.network import build_pass_network
from .analytics.ratings import DixonColes, Elo, rps, simulate_season
from .analytics.scouting import ScoutingModel
from .analytics.xg import XGModel
from .analytics.xt import ExpectedThreat
from .data.league import build_database


def _f(v, nd=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


class Platform:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not db_path.exists():
            build_database(db_path)
        t0 = time.time()
        con = sqlite3.connect(db_path)
        self.teams = pd.read_sql("SELECT * FROM teams", con).set_index("id", drop=False)
        self.players = pd.read_sql("SELECT * FROM players", con).set_index("id", drop=False)
        self.matches = pd.read_sql("SELECT * FROM matches", con).set_index("id", drop=False)
        self.apps = pd.read_sql("SELECT * FROM appearances", con)
        ev = pd.read_sql("SELECT * FROM events", con)
        con.close()
        self.timings = {"load": time.time() - t0}

        # ---------------------------------------------------------------- xG
        t = time.time()
        shots = ev[ev.type == "shot"]
        self.xg = XGModel().fit({
            "x": shots.x.values, "y": shots.y.values, "header": (shots.body_part == "head").values,
            "situation": shots.situation.values, "outcome": shots.outcome.values, "true_xg": shots.true_xg.values})
        ev["xg"] = np.nan
        ev.loc[shots.index, "xg"] = self.xg.predict(shots.x.values, shots.y.values,
                                                   (shots.body_part == "head").values, shots.situation.values)
        self.timings["xg"] = time.time() - t

        # ---------------------------------------------------------------- xT
        t = time.time()
        moves = ev[ev.type.isin(["pass", "carry"]) & ev.end_x.notna()]
        self.xt = ExpectedThreat().fit(
            {"x": moves.x.values, "y": moves.y.values, "end_x": moves.end_x.values, "end_y": moves.end_y.values,
             "success": (moves.outcome == "success").values},
            {"x": shots.x.values, "y": shots.y.values, "goal": (shots.outcome == "goal").values})
        ev["xt"] = np.nan
        ok = moves[moves.outcome == "success"]
        ev.loc[ok.index, "xt"] = self.xt.action_values(ok.x.values, ok.y.values, ok.end_x.values, ok.end_y.values)
        self.timings["xt"] = time.time() - t
        self.events = ev

        # ---------------------------------------------------------- team models
        t = time.time()
        self.played = self.matches[self.matches.played == 1].copy()
        self.remaining = self.matches[self.matches.played == 0].copy()
        self.current_md = int(self.played.matchday.max())
        xg_match = ev[ev.type == "shot"].groupby(["match_id", "team_id"]).xg.sum().unstack(fill_value=0)
        self.played["home_xg"] = [xg_match.loc[m, h] if h in xg_match.columns else 0
                                  for m, h in zip(self.played.id, self.played.home_id)]
        self.played["away_xg"] = [xg_match.loc[m, a] if a in xg_match.columns else 0
                                  for m, a in zip(self.played.id, self.played.away_id)]
        self.team_ids = self.teams.id.tolist()
        self.elo = Elo().run(self.team_ids, self.played.to_dict("records"))
        self.dc = self._fit_dc(self.played, "goals")
        self.dc_xg = self._fit_dc(self.played, "xg")
        self.timings["ratings"] = time.time() - t

        # ------------------------------------------------------------- scouting
        t = time.time()
        self.player_rows = self._player_features()
        self.scout = ScoutingModel().fit([r for r in self.player_rows if r["minutes"] >= 600])
        self.timings["scouting"] = time.time() - t
        self.timings["total"] = time.time() - t0
        print(f"[pitchiq] platform ready in {self.timings['total']:.2f}s "
              f"({len(ev):,} events, {len(shots):,} shots)")

    # ------------------------------------------------------------------ helpers
    def _fit_dc(self, df, kind):
        idx = {t: i for i, t in enumerate(self.team_ids)}
        hg = df.home_goals.values if kind == "goals" else df.home_xg.values
        ag = df.away_goals.values if kind == "goals" else df.away_xg.values
        return DixonColes(xi=0.02).fit(self.team_ids, df.home_id.map(idx).values, df.away_id.map(idx).values,
                                       hg, ag, df.matchday.values)

    def team_name(self, tid):
        return self.teams.loc[tid, "name"]

    def _player_features(self):
        ev = self.events
        mins = self.apps.groupby("player_id").minutes.sum()
        apps = self.apps.groupby("player_id").match_id.nunique()
        g = ev.groupby("player_id")
        shots = ev[ev.type == "shot"]
        passes = ev[ev.type == "pass"]
        carries = ev[ev.type == "carry"]
        okp = passes[passes.outcome == "success"]
        prog = okp[(okp.end_x - okp.x >= 12) & (okp.end_x > 60)]
        box = ev[ev.type.isin(["pass", "carry"]) & (ev.outcome == "success") & (ev.end_x > 102) &
                 (ev.end_y > 18) & (ev.end_y < 62) & ~((ev.x > 102) & (ev.y > 18) & (ev.y < 62))]
        cnt = lambda df: df.groupby("player_id").size()
        s_goals = cnt(shots[shots.outcome == "goal"])
        s_xg = shots.groupby("player_id").xg.sum()
        s_np_xg = shots[shots.situation != "penalty"].groupby("player_id").xg.sum()
        s_shots = cnt(shots)
        p_all, p_ok = cnt(passes), cnt(okp)
        c_all, c_ok = cnt(carries), cnt(carries[carries.outcome == "success"])
        xt = ev.groupby("player_id").xt.sum()
        tk, ic = cnt(ev[ev.type == "tackle"]), cnt(ev[ev.type == "interception"])
        prg, bx = cnt(prog), cnt(box)
        avg_x = ev[ev.type.isin(["pass", "carry"])].groupby("player_id").x.mean()
        assists = self._assists()
        rows = []
        for pid, p in self.players.iterrows():
            m = float(mins.get(pid, 0))
            per = 90 / m if m else 0
            get = lambda s: float(s.get(pid, 0))
            rows.append({
                "player_id": int(pid), "name": p["name"], "team_id": int(p.team_id),
                "team": self.team_name(p.team_id), "position": p.position, "age": int(p.age),
                "minutes": int(m), "apps": int(apps.get(pid, 0)),
                "goals": int(get(s_goals)), "assists": int(assists.get(pid, 0)), "xg": get(s_xg),
                "npxg": get(s_np_xg), "shots": int(get(s_shots)), "xt": get(xt),
                "goals_p90": get(s_goals) * per, "xg_p90": get(s_xg) * per, "shots_p90": get(s_shots) * per,
                "xg_per_shot": get(s_xg) / get(s_shots) if get(s_shots) else 0.0,
                "passes_p90": get(p_all) * per, "pass_pct": get(p_ok) / get(p_all) * 100 if get(p_all) else 0.0,
                "prog_passes_p90": get(prg) * per, "xt_p90": get(xt) * per, "box_entries_p90": get(bx) * per,
                "carries_p90": get(c_all) * per,
                "dribble_pct": get(c_ok) / get(c_all) * 100 if get(c_all) else 0.0,
                "tackles_p90": get(tk) * per, "interceptions_p90": get(ic) * per,
                "avg_x": float(avg_x.get(pid, 60.0)),
                "rating_truth": {k: float(p[k]) for k in ("passing", "finishing", "dribbling", "defending")},
            })
        return rows

    def _assists(self):
        ev = self.events
        goals = ev[(ev.type == "shot") & (ev.outcome == "goal") & (ev.situation != "penalty")]
        okp = ev[(ev.type == "pass") & (ev.outcome == "success")]
        last = okp.groupby(["match_id", "possession", "receiver_id"]).player_id.last()
        out = {}
        for (mid, pos, shooter) in zip(goals.match_id, goals.possession, goals.player_id):
            a = last.get((mid, pos, shooter))
            if a is not None and a != shooter:
                out[a] = out.get(a, 0) + 1
        return out

    # ------------------------------------------------------------------ league
    @lru_cache(maxsize=1)
    def league_table(self):
        rows = {t: dict(team_id=int(t), name=self.team_name(t), short=self.teams.loc[t, "short"], p=0, w=0, d=0,
                        l=0, gf=0, ga=0, xgf=0.0, xga=0.0, pts=0, form=[]) for t in self.team_ids}
        for m in self.played.sort_values("matchday").itertuples():
            for t, o, gf, ga, xf, xa in ((m.home_id, m.away_id, m.home_goals, m.away_goals, m.home_xg, m.away_xg),
                                         (m.away_id, m.home_id, m.away_goals, m.home_goals, m.away_xg, m.home_xg)):
                r = rows[t]
                r["p"] += 1; r["gf"] += gf; r["ga"] += ga; r["xgf"] += xf; r["xga"] += xa
                res = "W" if gf > ga else "D" if gf == ga else "L"
                r[res.lower()] += 1
                r["pts"] += {"W": 3, "D": 1, "L": 0}[res]
                r["form"].append(res)
        xpts = self._expected_points()
        out = []
        for r in rows.values():
            r["form"] = r["form"][-5:]
            r["gd"] = r["gf"] - r["ga"]
            r["xgd"] = round(r["xgf"] - r["xga"], 2)
            r["xgf"], r["xga"] = round(r["xgf"], 2), round(r["xga"], 2)
            r["xpts"] = round(xpts[r["team_id"]], 1)
            r["elo"] = round(self.elo.ratings[r["team_id"]], 0)
            out.append(r)
        out.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
        for i, r in enumerate(out, 1):
            r["position"] = i
        return out

    def _expected_points(self):
        """xPts: simulate each played match 2,000 times from its actual shot list (shot-level xG)."""
        rng = np.random.default_rng(3)
        shots = self.events[self.events.type == "shot"]
        xp = {t: 0.0 for t in self.team_ids}
        for (mid), grp in shots.groupby("match_id"):
            m = self.matches.loc[mid]
            hx = grp[grp.team_id == m.home_id].xg.values
            ax = grp[grp.team_id == m.away_id].xg.values
            hg = (rng.random((2000, len(hx))) < hx).sum(1)
            ag = (rng.random((2000, len(ax))) < ax).sum(1)
            xp[m.home_id] += float(np.mean(np.where(hg > ag, 3, np.where(hg == ag, 1, 0))))
            xp[m.away_id] += float(np.mean(np.where(ag > hg, 3, np.where(hg == ag, 1, 0))))
        return xp

    def overview(self):
        ev = self.events
        shots = ev[ev.type == "shot"]
        table = self.league_table()
        top_scorers = sorted(self.player_rows, key=lambda r: (-r["goals"], -r["xg"]))[:8]
        over = sorted(self.player_rows, key=lambda r: -(r["goals"] - r["npxg"]))[:5]
        return {
            "kpis": {
                "matches_played": int(len(self.played)), "matchday": self.current_md,
                "total_matchdays": int(self.matches.matchday.max()),
                "events": int(len(ev)), "shots": int(len(shots)),
                "goals": int((shots.outcome == "goal").sum()),
                "goals_per_match": round(float(self.played.home_goals.sum() + self.played.away_goals.sum()) /
                                         len(self.played), 2),
                "home_win_pct": round(float((self.played.home_goals > self.played.away_goals).mean()) * 100, 1),
                "xg_auc": round(self.xg.report["metrics"]["model"]["auc"], 3),
                "pipeline_seconds": round(self.timings["total"], 2),
            },
            "leader": table[0],
            "top_scorers": [{k: r[k] for k in ("player_id", "name", "team", "goals", "assists", "xg", "shots",
                                                "minutes")} for r in top_scorers],
            "overperformers": [{"player_id": r["player_id"], "name": r["name"], "team": r["team"],
                                "goals": r["goals"], "npxg": round(r["npxg"], 2)} for r in over],
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }

    def elo_history(self):
        return [{"team_id": int(t), "name": self.team_name(t), "short": self.teams.loc[t, "short"],
                 "history": self.elo.history[t]} for t in self.team_ids]

    def ratings(self):
        goals = {r["team_id"]: r for r in self.dc.ratings_table()}
        xg = {r["team_id"]: r for r in self.dc_xg.ratings_table()}
        rows = []
        for t in self.team_ids:
            rows.append({"team_id": int(t), "name": self.team_name(t), "short": self.teams.loc[t, "short"],
                         "attack": goals[t]["attack"], "defence": goals[t]["defence"],
                         "xg_attack": xg[t]["attack"], "xg_defence": xg[t]["defence"],
                         "elo": self.elo.ratings[t],
                         "true_attack": float(self.teams.loc[t, "true_attack"]),
                         "true_defence": float(self.teams.loc[t, "true_defence"])})
        df = pd.DataFrame(rows)
        corr = lambda a, b: round(float(np.corrcoef(df[a], df[b])[0, 1]), 3)
        return {"teams": rows,
                "params": {"home_advantage": float(self.dc.home), "rho": float(self.dc.rho),
                           "intercept": float(self.dc.mu), "xi": self.dc.xi, "converged": self.dc.converged,
                           "iterations": self.dc.n_iter},
                "truth_recovery": {
                    "goals_model": {"attack_r": corr("attack", "true_attack"),
                                    "defence_r": corr("defence", "true_defence")},
                    "xg_model": {"attack_r": corr("xg_attack", "true_attack"),
                                 "defence_r": corr("xg_defence", "true_defence")}}}

    # ------------------------------------------------------------------ teams
    def team_detail(self, tid: int):
        tbl = {r["team_id"]: r for r in self.league_table()}
        ev = self.events
        mine = ev[ev.team_id == tid]
        shots = mine[mine.type == "shot"]
        conc = ev[(ev.type == "shot") & (ev.team_id != tid) &
                  ev.match_id.isin(self.played[(self.played.home_id == tid) | (self.played.away_id == tid)].id)]
        players = sorted([r for r in self.player_rows if r["team_id"] == tid], key=lambda r: -r["minutes"])
        fixtures = []
        for m in self.matches[(self.matches.home_id == tid) | (self.matches.away_id == tid)].itertuples():
            fixtures.append(self._match_row(m))
        heat, _, _ = np.histogram2d(mine[mine.type.isin(["pass", "carry"])].x, mine[mine.type.isin(["pass", "carry"])].y,
                                    bins=[24, 16], range=[[0, 120], [0, 80]])
        return {
            "team": {"id": int(tid), "name": self.team_name(tid), "short": self.teams.loc[tid, "short"]},
            "table": tbl[tid],
            "shots_for": self._shot_list(shots), "shots_against": self._shot_list(conc),
            "players": players, "fixtures": fixtures,
            "heatmap": (heat.T / max(heat.max(), 1)).round(3).tolist(),
            "elo": self.elo.history[tid],
        }

    def _shot_list(self, shots):
        return [{"id": int(s.id), "match_id": int(s.match_id), "player_id": int(s.player_id),
                 "player": self.players.loc[s.player_id, "name"], "team_id": int(s.team_id),
                 "minute": int(s.second // 60), "x": s.x, "y": s.y, "xg": round(float(s.xg), 3),
                 "outcome": s.outcome, "body_part": s.body_part, "situation": s.situation}
                for s in shots.itertuples()]

    # ---------------------------------------------------------------- matches
    def _match_row(self, m):
        row = {"id": int(m.id), "matchday": int(m.matchday), "home_id": int(m.home_id), "away_id": int(m.away_id),
               "home": self.team_name(m.home_id), "away": self.team_name(m.away_id),
               "home_short": self.teams.loc[m.home_id, "short"], "away_short": self.teams.loc[m.away_id, "short"],
               "played": bool(m.played)}
        if m.played:
            pm = self.played.loc[m.id]
            row.update(home_goals=int(m.home_goals), away_goals=int(m.away_goals),
                       home_xg=round(float(pm.home_xg), 2), away_xg=round(float(pm.away_xg), 2))
        else:
            p = self.dc.probs(m.home_id, m.away_id)
            row.update(prob_home=round(float(p[0]), 3), prob_draw=round(float(p[1]), 3), prob_away=round(float(p[2]), 3))
        return row

    def match_list(self, matchday: Optional[int] = None, team_id: Optional[int] = None):
        df = self.matches
        if matchday:
            df = df[df.matchday == matchday]
        if team_id:
            df = df[(df.home_id == team_id) | (df.away_id == team_id)]
        return [self._match_row(m) for m in df.itertuples()]

    @lru_cache(maxsize=64)
    def match_detail(self, mid: int):
        m = self.matches.loc[mid]
        if not m.played:
            return None
        ev = self.events[self.events.match_id == mid]
        shots = ev[ev.type == "shot"].sort_values("second")
        pl = self.players.to_dict("index")
        side = {}
        for key, tid in (("home", m.home_id), ("away", m.away_id)):
            te = ev[ev.team_id == tid]
            ps = te[te.type == "pass"]
            ok = ps[ps.outcome == "success"]
            sh = te[te.type == "shot"]
            side[key] = {
                "team_id": int(tid), "name": self.team_name(tid), "short": self.teams.loc[tid, "short"],
                "goals": int((sh.outcome == "goal").sum()), "xg": round(float(sh.xg.sum()), 2),
                "shots": int(len(sh)), "on_target": int(sh.outcome.isin(["goal", "saved"]).sum()),
                "passes": int(len(ps)), "pass_pct": round(float(len(ok) / max(len(ps), 1) * 100), 1),
                "xt": round(float(te.xt.sum()), 2),
                "tackles": int((te.type == "tackle").sum()), "interceptions": int((te.type == "interception").sum()),
                "network": build_pass_network(ok[["player_id", "receiver_id", "x", "y", "end_x", "end_y"]]
                                              .to_dict("records"), pl),
            }
        tot = side["home"]["passes"] + side["away"]["passes"]
        side["home"]["possession"] = round(side["home"]["passes"] / tot * 100, 1)
        side["away"]["possession"] = round(100 - side["home"]["possession"], 1)

        # cumulative xG timeline & a post-match "deserved result" via shot-level simulation
        timeline = []
        cum = {int(m.home_id): 0.0, int(m.away_id): 0.0}
        for s in shots.itertuples():
            cum[int(s.team_id)] += float(s.xg)
            timeline.append({"minute": round(s.second / 60, 2), "team_id": int(s.team_id), "xg": round(float(s.xg), 3),
                             "home_cum": round(cum[int(m.home_id)], 3), "away_cum": round(cum[int(m.away_id)], 3),
                             "goal": s.outcome == "goal", "player": pl[s.player_id]["name"]})
        rng = np.random.default_rng(int(mid))
        hx = shots[shots.team_id == m.home_id].xg.values
        ax = shots[shots.team_id == m.away_id].xg.values
        hg = (rng.random((10000, len(hx))) < hx).sum(1)
        ag = (rng.random((10000, len(ax))) < ax).sum(1)
        # xT flow by 5-minute bucket (momentum chart)
        buckets = np.arange(0, 100, 5)
        mom = []
        for b in buckets:
            seg = ev[(ev.second >= b * 60) & (ev.second < (b + 5) * 60)]
            hv = float(seg[seg.team_id == m.home_id].xt.clip(lower=0).sum())
            av = float(seg[seg.team_id == m.away_id].xt.clip(lower=0).sum())
            mom.append({"minute": int(b), "home": round(hv, 3), "away": round(av, 3)})
        return {
            "match": self._match_row(m), "home": side["home"], "away": side["away"],
            "shots": self._shot_list(shots), "timeline": timeline, "momentum": mom,
            "deserved": {"home": round(float((hg > ag).mean()), 3), "draw": round(float((hg == ag).mean()), 3),
                         "away": round(float((hg < ag).mean()), 3)},
        }

    # ------------------------------------------------------------- prediction
    def predict(self, home: int, away: int):
        M = self.dc.score_matrix(home, away)
        lam, mu = self.dc.rates(home, away)
        Mx = self.dc_xg.score_matrix(home, away)
        flat = [(i, j, float(M[i, j])) for i in range(7) for j in range(7)]
        flat.sort(key=lambda t: -t[2])
        p = self.dc.probs(home, away)
        px = self.dc_xg.probs(home, away)
        pe = self.elo.probs(home, away)
        ens = (p + px + pe) / 3
        return {
            "home": {"id": home, "name": self.team_name(home), "short": self.teams.loc[home, "short"]},
            "away": {"id": away, "name": self.team_name(away), "short": self.teams.loc[away, "short"]},
            "expected_goals": {"home": lam, "away": mu},
            "models": {
                "dixon_coles": dict(zip(("home", "draw", "away"), map(float, p))),
                "xg_poisson": dict(zip(("home", "draw", "away"), map(float, px))),
                "elo": dict(zip(("home", "draw", "away"), map(float, pe))),
                "ensemble": dict(zip(("home", "draw", "away"), map(float, ens))),
            },
            "score_matrix": M[:7, :7].round(4).tolist(),
            "xg_score_matrix": Mx[:7, :7].round(4).tolist(),
            "top_scores": [{"home": i, "away": j, "p": pr} for i, j, pr in flat[:6]],
            "btts": float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]),
            "over_2_5": float(sum(M[i, j] for i in range(11) for j in range(11) if i + j > 2)),
            "clean_sheet": {"home": float(M[:, 0].sum()), "away": float(M[0, :].sum())},
        }

    @lru_cache(maxsize=8)
    def season_simulation(self, n_sims: int = 10000, model: str = "dixon_coles", seed: int = 0):
        t = time.time()
        mdl = self.dc if model == "dixon_coles" else self.dc_xg
        played = list(zip(self.played.home_id, self.played.away_id, self.played.home_goals, self.played.away_goals))
        remaining = list(zip(self.remaining.home_id, self.remaining.away_id))
        res = simulate_season(mdl, self.team_ids, played, remaining, n_sims=n_sims, seed=seed)
        for r in res:
            r["name"] = self.team_name(r["team_id"])
            r["short"] = self.teams.loc[r["team_id"], "short"]
        res.sort(key=lambda r: r["expected_position"])
        return {"n_sims": n_sims, "model": model, "remaining_fixtures": len(remaining),
                "seconds": round(time.time() - t, 3), "teams": res}

    @lru_cache(maxsize=1)
    def backtest(self, start_md: int = 8):
        """Walk-forward: refit on matchdays < k, forecast matchday k. Compare RPS & accuracy."""
        out = {"elo": [], "dixon_coles": [], "xg_poisson": [], "ensemble": [], "uniform": []}
        per_md = []
        for k in range(start_md, self.current_md + 1):
            train = self.played[self.played.matchday < k]
            test = self.played[self.played.matchday == k]
            dc = self._fit_dc(train, "goals")
            dcx = self._fit_dc(train, "xg")
            elo = Elo().run(self.team_ids, train.to_dict("records"))
            md_scores = {key: [] for key in out}
            for m in test.itertuples():
                o = 0 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals else 2
                preds = {"dixon_coles": dc.probs(m.home_id, m.away_id),
                         "xg_poisson": dcx.probs(m.home_id, m.away_id),
                         "elo": elo.probs(m.home_id, m.away_id), "uniform": np.full(3, 1 / 3)}
                preds["ensemble"] = (preds["dixon_coles"] + preds["xg_poisson"] + preds["elo"]) / 3
                for key, pr in preds.items():
                    s = rps(pr, o)
                    out[key].append((s, int(np.argmax(pr) == o), float(-np.log(pr[o]))))
                    md_scores[key].append(s)
            per_md.append({"matchday": k, **{key: float(np.mean(v)) for key, v in md_scores.items()}})
        summary = {key: {"rps": float(np.mean([v[0] for v in vals])), "accuracy": float(np.mean([v[1] for v in vals])),
                         "log_loss": float(np.mean([v[2] for v in vals])), "n": len(vals)}
                   for key, vals in out.items()}
        return {"summary": summary, "per_matchday": per_md, "start_matchday": start_md}

    # ---------------------------------------------------------------- players
    def player_list(self, team_id=None, position=None, min_minutes=0, sort="xt_p90", limit=500):
        rows = [r for r in self.player_rows if r["minutes"] >= min_minutes
                and (team_id is None or r["team_id"] == team_id) and (position is None or r["position"] == position)]
        rows.sort(key=lambda r: -r.get(sort, 0))
        return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "rating_truth"}
                for r in rows[:limit]]

    def player_detail(self, pid: int):
        row = next((r for r in self.player_rows if r["player_id"] == pid), None)
        if row is None:
            return None
        ev = self.events[self.events.player_id == pid]
        shots = ev[ev.type == "shot"]
        touches = ev[ev.type.isin(["pass", "carry"])]
        heat, _, _ = np.histogram2d(touches.x, touches.y, bins=[24, 16], range=[[0, 120], [0, 80]])
        top_xt = ev[ev.xt.notna()].nlargest(40, "xt")
        out = {"player": {k: v for k, v in row.items()}, "shots": self._shot_list(shots),
               "heatmap": (heat.T / max(heat.max(), 1)).round(3).tolist(),
               "top_actions": [{"type": a.type, "x": a.x, "y": a.y, "end_x": a.end_x, "end_y": a.end_y,
                                "xt": round(float(a.xt), 4)} for a in top_xt.itertuples()]}
        if pid in self.scout.index:
            out["profile"] = self.scout.profile(pid)
            out["similar"] = self.scout.similar(pid)
        return out

    def xt_leaders(self, n=15):
        rows = sorted([r for r in self.player_rows if r["minutes"] >= 600], key=lambda r: -r["xt_p90"])[:n]
        return [{"player_id": r["player_id"], "name": r["name"], "team": r["team"], "position": r["position"],
                 "xt": round(r["xt"], 3), "xt_p90": round(r["xt_p90"], 4), "minutes": r["minutes"]} for r in rows]
