"""
Stochastic match engine.

Generates StatsBomb-style event data (passes, carries, shots, defensive actions)
by simulating every possession as a Markov chain over pitch locations.

Why simulate instead of scraping?
  * Ground truth is known. Every team has a latent attack/defence strength and every
    shot has a *true* scoring probability, so we can prove our models recover reality
    (e.g. "fitted Dixon-Coles attack ratings correlate r=0.9 with the hidden truth").
  * Fully offline, deterministic (seeded) and license-free - perfect for a live demo.
  * The same engine powers the WebSocket live-match stream.

Coordinate system (same as StatsBomb): pitch is 120 x 80, every event is recorded
from the acting team's perspective, attacking left -> right, goal centre at (120, 40).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator

PITCH_X, PITCH_Y = 120.0, 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
POST_L, POST_R = 36.0, 44.0
MATCH_SECONDS = 90 * 60

# Formation anchors (4-3-3), attacking to the right.
FORMATION = [
    ("GK", 8, 40), ("RB", 40, 68), ("CB", 30, 50), ("CB", 30, 30), ("LB", 40, 12),
    ("DM", 50, 40), ("CM", 62, 54), ("CM", 62, 26),
    ("RW", 88, 68), ("LW", 88, 12), ("ST", 98, 40),
]


ATTACK_WEIGHT = {"ST": 2.0, "RW": 1.8, "LW": 1.8, "CM": 1.3, "DM": 0.4, "CB": 0.5, "RB": 0.4, "LB": 0.4}


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """Distance to goal centre and the visible goal-mouth angle (radians)."""
    dx, dy = GOAL_X - x, GOAL_Y - y
    dist = math.hypot(dx, dy)
    a1 = math.atan2(POST_L - y, dx)
    a2 = math.atan2(POST_R - y, dx)
    angle = abs(a2 - a1)
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return dist, angle


def true_goal_probability(dist, angle, header, situation, finishing, gk_skill) -> float:
    """The hidden 'physics' of finishing. Our xG model never sees this function."""
    if situation == "penalty":
        return min(0.95, 0.76 + 0.04 * finishing - 0.03 * gk_skill)
    z = (-0.60 - 0.13 * dist + 0.90 * angle
         - 0.95 * header
         + (0.40 if situation == "counter" else 0.0)
         - (0.30 if situation == "free_kick" else 0.0)
         + 0.30 * finishing - 0.20 * gk_skill)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class PlayerState:
    id: int
    name: str
    position: str
    ax: float
    ay: float
    passing: float
    finishing: float
    dribbling: float
    defending: float


@dataclass
class TeamState:
    id: int
    attack: float
    defence: float
    directness: float
    lineup: list[PlayerState] = field(default_factory=list)

    @property
    def gk(self) -> PlayerState:
        return self.lineup[0]


@dataclass
class MatchEvent:
    second: int
    team_id: int
    player_id: int
    type: str                 # pass | carry | shot | interception | tackle
    x: float
    y: float
    end_x: float | None = None
    end_y: float | None = None
    outcome: str = "success"  # success | fail | goal | saved | off_target | blocked
    receiver_id: int | None = None
    possession: int = 0
    body_part: str | None = None
    situation: str | None = None
    true_xg: float | None = None

    @property
    def minute(self) -> int:
        return self.second // 60


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class MatchSimulator:
    """Simulates one match, yielding events in chronological order."""

    def __init__(self, home: TeamState, away: TeamState, rng: random.Random, home_adv: float = 0.12):
        self.home, self.away, self.rng, self.home_adv = home, away, rng, home_adv
        self.clock = 0
        self.score = {home.id: 0, away.id: 0}
        self.possession_id = 0
        self.end_second = MATCH_SECONDS + rng.randint(120, 360)

    # ---- helpers -------------------------------------------------------------
    def _boost(self, team: TeamState) -> float:
        return self.home_adv if team is self.home else 0.0

    def _other(self, team: TeamState) -> TeamState:
        return self.away if team is self.home else self.home

    def _pick_receiver(self, team, holder, tx, ty, ball_x):
        r = self.rng
        shift = (ball_x - 60) * 0.35
        best, weights = [], []
        for p in team.lineup:
            if p is holder:
                continue
            px = _clip(p.ax + shift, 4, 116)
            d2 = (px - tx) ** 2 + (p.ay - ty) ** 2
            w = math.exp(-d2 / (2 * 16.0 ** 2)) + 1e-4
            if p.position == "GK":
                w *= 0.15
            best.append((p, px))
            weights.append(w)
        (p, px), = r.choices(best, weights=weights, k=1)
        return p, _clip(px + r.gauss(0, 6), 1, 119), _clip(p.ay + r.gauss(0, 6), 1, 79)

    def _nearest_defender(self, team, x, y):
        # opponent coordinates are mirrored
        mx, my = PITCH_X - x, PITCH_Y - y
        # closest players compete for the ball; better defenders win it more often
        cands = sorted(team.lineup[1:], key=lambda p: (p.ax - mx) ** 2 + (p.ay - my) ** 2)[:4]
        w = [math.exp(1.2 * p.defending) / (1 + ((p.ax - mx) ** 2 + (p.ay - my) ** 2) / 400) for p in cands]
        return self.rng.choices(cands, weights=w, k=1)[0]

    # ---- main loop -----------------------------------------------------------
    def run(self) -> Iterator[MatchEvent]:
        team = self.home if self.rng.random() < 0.5 else self.away
        x, y = 60.0, 40.0
        situation_start = "kickoff"
        while self.clock < self.end_second:
            self.possession_id += 1
            result = yield from self._possession(team, x, y, situation_start)
            kind, x, y = result
            team = self._other(team)
            if kind == "goal":
                x, y, situation_start = 60.0, 40.0, "kickoff"
            elif kind == "shot_over":
                x, y, situation_start = 6.0, 40.0, "goal_kick"
            else:  # turnover: new team starts where it won the ball (mirrored)
                x, y, situation_start = PITCH_X - x, PITCH_Y - y, "regain"
            self.clock += self.rng.randint(3, 10)

    def _possession(self, team: TeamState, x, y, start_kind):
        r = self.rng
        opp = self._other(team)
        boost = self._boost(team)
        start_x = x
        holder = team.gk if start_kind == "goal_kick" else min(
            team.lineup, key=lambda p: (p.ax - x) ** 2 + (p.ay - y) ** 2)
        steps = 0
        while steps < 30:
            steps += 1
            self.clock += r.randint(2, 5)
            dist, angle = shot_geometry(x, y)

            # 1) Shoot?
            if dist < 35 and x > 80:
                p_shot = 0.23 * math.exp(-(dist - 12) / 19.0) * (1 + 0.6 * (team.attack - opp.defence))
                if r.random() < p_shot:
                    sit = "counter" if (start_x < 55 and steps <= 5) else "open_play"
                    if start_kind == "corner":
                        sit = "corner"
                    header = r.random() < (0.55 if sit == "corner" else 0.10 if dist < 16 else 0.01)
                    shooter = holder
                    if r.random() < 0.7:   # lay-offs / cut-backs: a supporting attacker arrives to shoot
                        pool = [p for p in team.lineup if p.position in ATTACK_WEIGHT]
                        shooter = r.choices(pool, weights=[ATTACK_WEIGHT[p.position] for p in pool], k=1)[0]
                    ev = self._shot(team, opp, shooter, x, y, header, sit)
                    yield ev
                    if ev.outcome == "goal":
                        return ("goal", x, y)
                    return ("shot_over", x, y) if ev.outcome != "saved" or r.random() < 0.6 else ("turnover", 112.0, 40.0)

            # 2) Carry / dribble
            if r.random() < 0.13 and holder.position != "GK":
                cdx = abs(r.gauss(7, 4))
                ex, ey = _clip(x + cdx, 1, 119), _clip(y + r.gauss(0, 4), 1, 79)
                p_ok = 1 / (1 + math.exp(-(1.6 + 0.5 * holder.dribbling - 0.45 * opp.defence + boost - 0.012 * max(0, ex - 70) * 2)))
                ok = r.random() < p_ok
                yield MatchEvent(self.clock, team.id, holder.id, "carry", x, y, ex, ey,
                                 "success" if ok else "fail", possession=self.possession_id)
                if not ok:
                    d = self._nearest_defender(opp, ex, ey)
                    yield MatchEvent(self.clock, opp.id, d.id, "tackle", PITCH_X - ex, PITCH_Y - ey,
                                     possession=self.possession_id)
                    return ("turnover", ex, ey)
                x, y = ex, ey
                continue

            # 3) Pass
            fwd = 5 + 6 * team.directness + (4 if x < 40 else 0)
            tx = x + r.gauss(fwd, 14)
            ty = y + r.gauss(0, 16)
            rec, ex, ey = self._pick_receiver(team, holder, tx, ty, x)
            length = math.hypot(ex - x, ey - y)
            z = (3.1 + 0.45 * holder.passing + 0.55 * team.attack - 0.60 * opp.defence + boost
                 - 0.032 * length - 0.022 * max(0, ex - 75) - 0.14 * max(0, ex - 100)
                 - 0.06 * max(0, ex - 108))
            ok = r.random() < 1 / (1 + math.exp(-z))
            if not ok and ex > 106 and r.random() < 0.28:
                # ball out for a corner - same team retains possession
                yield MatchEvent(self.clock, team.id, holder.id, "pass", x, y, ex, ey, "fail",
                                 rec.id, possession=self.possession_id)
                self.clock += r.randint(15, 30)
                cy = 0.5 if r.random() < 0.5 else 79.5
                taker = max(team.lineup, key=lambda p: p.passing)
                tx2, ty2 = 120 - abs(r.gauss(7, 3)), 40 + r.gauss(0, 7)
                target = r.choice([p for p in team.lineup if p.position in ("CB", "ST", "CM", "DM")])
                self.possession_id += 1
                hit = r.random() < 0.42
                yield MatchEvent(self.clock, team.id, taker.id, "pass", 119.5, cy, tx2, ty2,
                                 "success" if hit else "fail", target.id, possession=self.possession_id,
                                 situation="corner")
                if not hit:
                    d = self._nearest_defender(opp, tx2, ty2)
                    yield MatchEvent(self.clock, opp.id, d.id, "interception", PITCH_X - tx2, PITCH_Y - ty2,
                                     possession=self.possession_id)
                    return ("turnover", tx2, ty2)
                holder, x, y, start_kind, start_x, steps = target, tx2, ty2, "corner", tx2, 0
                continue

            yield MatchEvent(self.clock, team.id, holder.id, "pass", x, y, ex, ey,
                             "success" if ok else "fail", rec.id, possession=self.possession_id,
                             situation=start_kind if steps == 1 else None)
            if not ok:
                d = self._nearest_defender(opp, ex, ey)
                yield MatchEvent(self.clock, opp.id, d.id, "interception", PITCH_X - ex, PITCH_Y - ey,
                                 possession=self.possession_id)
                return ("turnover", ex, ey)
            # Foul in the box -> penalty (rare)
            if ex > 102 and 18 < ey < 62 and r.random() < 0.006:
                self.clock += 60
                ev = self._shot(team, opp, rec, 108.0, 40.0, False, "penalty")
                yield ev
                return ("goal", 108, 40) if ev.outcome == "goal" else ("shot_over", 108, 40)
            # Free kick in shooting range (rare)
            if 85 < ex < 100 and 20 < ey < 60 and r.random() < 0.012:
                self.clock += 40
                taker = max(team.lineup, key=lambda p: p.finishing + p.passing)
                ev = self._shot(team, opp, taker, ex, ey, False, "free_kick")
                yield ev
                return ("goal", ex, ey) if ev.outcome == "goal" else ("shot_over", ex, ey)
            holder, x, y = rec, ex, ey
            if start_kind == "corner" and steps > 2:
                start_kind = "open_play"
        return ("turnover", x, y)

    def _shot(self, team, opp, shooter, x, y, header, situation) -> MatchEvent:
        r = self.rng
        dist, angle = shot_geometry(x, y)
        p = true_goal_probability(dist, angle, header, situation, shooter.finishing, opp.gk.defending)
        u = r.random()
        if u < p:
            outcome = "goal"
            self.score[team.id] += 1
        else:
            rest = (u - p) / (1 - p)
            outcome = "saved" if rest < 0.45 else "blocked" if rest < 0.70 else "off_target"
        return MatchEvent(self.clock, team.id, shooter.id, "shot", x, y, GOAL_X,
                          _clip(40 + r.gauss(0, 5), 30, 50), outcome,
                          possession=self.possession_id, body_part="head" if header else "foot",
                          situation=situation, true_xg=p)
