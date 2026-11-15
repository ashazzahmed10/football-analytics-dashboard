"""
Expected Threat (xT) - a Markov-chain model of ball progression (Karun Singh, 2018).

The pitch is split into a 16 x 12 grid. For each cell z we estimate from data:
    s(z)        probability a player in z shoots
    g(z)        probability a shot from z scores
    m(z)        probability a player in z moves the ball (pass/carry)
    T(z -> z')  probability a move from z ends successfully in z'

xT is the fixed point of the Bellman-style equation
    xT(z) = s(z) * g(z) + m(z) * sum_z' T(z -> z') * xT(z')
solved by value iteration. Each successful action is then credited with
xT(end) - xT(start): how much it increased the chance of scoring soon.
"""
from __future__ import annotations

import numpy as np

NX, NY = 16, 12
L, W = 120.0, 80.0


def cell_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx = np.clip((np.asarray(x) / L * NX).astype(int), 0, NX - 1)
    cy = np.clip((np.asarray(y) / W * NY).astype(int), 0, NY - 1)
    return cy * NX + cx


class ExpectedThreat:
    def fit(self, moves: dict[str, np.ndarray], shots: dict[str, np.ndarray], tol=1e-6, max_iter=100):
        n = NX * NY
        start = cell_index(moves["x"], moves["y"])
        end = cell_index(moves["end_x"], moves["end_y"])
        ok = moves["success"]
        shot_cells = cell_index(shots["x"], shots["y"])
        goals = shots["goal"]

        move_count = np.bincount(start, minlength=n).astype(float)
        shot_count = np.bincount(shot_cells, minlength=n).astype(float)
        goal_count = np.bincount(shot_cells, weights=goals.astype(float), minlength=n)

        total = move_count + shot_count
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(total > 0, shot_count / total, 0.0)
            m = np.where(total > 0, move_count / total, 0.0)
            g = np.where(shot_count > 0, goal_count / shot_count, 0.0)

        T = np.zeros((n, n))
        np.add.at(T, (start[ok], end[ok]), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(move_count[:, None] > 0, T / move_count[:, None], 0.0)

        xt = np.zeros(n)
        self.snapshots, self.deltas = [], []
        for it in range(max_iter):
            new = s * g + m * (T @ xt)
            delta = float(np.max(np.abs(new - xt)))
            xt = new
            self.deltas.append(delta)
            if it < 8 or delta < tol:
                self.snapshots.append(xt.reshape(NY, NX).round(5).tolist())
            if delta < tol:
                break
        self.iterations = it + 1
        self.grid = xt
        self.components = {"shot_prob": s, "goal_prob": g, "move_prob": m}
        return self

    def value(self, x, y) -> np.ndarray:
        return self.grid[cell_index(x, y)]

    def action_values(self, x, y, end_x, end_y) -> np.ndarray:
        return self.value(end_x, end_y) - self.value(x, y)

    def to_dict(self):
        return {
            "nx": NX, "ny": NY, "grid": self.grid.reshape(NY, NX).round(5).tolist(),
            "iterations": self.iterations, "deltas": self.deltas,
            "snapshots": self.snapshots,
            "shot_prob": self.components["shot_prob"].reshape(NY, NX).round(4).tolist(),
            "goal_prob": self.components["goal_prob"].reshape(NY, NX).round(4).tolist(),
            "move_prob": self.components["move_prob"].reshape(NY, NX).round(4).tolist(),
        }
