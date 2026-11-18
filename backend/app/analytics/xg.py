"""
Expected Goals (xG) model - logistic regression implemented from scratch.

Optimiser: Newton-Raphson / Iteratively Re-weighted Least Squares (IRLS) with an
L2 penalty. Each iteration solves   (X^T W X + lambda*I) delta = X^T (y - p) - lambda*beta
which converges quadratically (typically < 10 iterations) - far faster than SGD.

Evaluation: log-loss, Brier score, ROC-AUC (Mann-Whitney U), and a reliability
(calibration) curve. Because the data comes from a simulator we also know every
shot's *true* probability, so we report the "oracle" log-loss as a lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.engine import shot_geometry

FEATURES = ["distance", "angle", "distance_x_angle", "log_distance", "header",
            "counter", "corner", "free_kick"]


def featurize(x: np.ndarray, y: np.ndarray, header: np.ndarray, situation: np.ndarray) -> np.ndarray:
    geo = np.array([shot_geometry(a, b) for a, b in zip(x, y)], dtype=float).reshape(-1, 2)
    d, ang = geo[:, 0], geo[:, 1]
    return np.column_stack([
        d, ang, d * ang, np.log1p(d), header.astype(float),
        (situation == "counter").astype(float), (situation == "corner").astype(float),
        (situation == "free_kick").astype(float),
    ])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, p) -> float:
    """AUC via the Mann-Whitney U statistic with average ranks for ties."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y, p, points: int = 50):
    th = np.quantile(p, np.linspace(0, 1, points))[::-1]
    P, N = y.sum(), len(y) - y.sum()
    out = [{"fpr": 0.0, "tpr": 0.0}]
    for t in th:
        pred = p >= t
        out.append({"fpr": float((pred & (y == 0)).sum() / N), "tpr": float((pred & (y == 1)).sum() / P)})
    out.append({"fpr": 1.0, "tpr": 1.0})
    return out


def calibration(y, p, bins: int = 10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append({"predicted": float(p[m].mean()), "observed": float(y[m].mean()), "n": int(m.sum())})
    return out


@dataclass
class LogisticRegressionIRLS:
    l2: float = 1.0
    max_iter: int = 50
    tol: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        beta = np.zeros(Xs.shape[1])
        reg = self.l2 * np.eye(Xs.shape[1])
        reg[0, 0] = 0.0  # never penalise the intercept
        self.history_ = []
        for it in range(self.max_iter):
            p = sigmoid(Xs @ beta)
            W = p * (1 - p)
            grad = Xs.T @ (y - p) - reg @ beta
            H = (Xs * W[:, None]).T @ Xs + reg
            step = np.linalg.solve(H, grad)
            beta += step
            self.history_.append(log_loss(y, sigmoid(Xs @ beta)))
            if np.max(np.abs(step)) < self.tol:
                break
        self.beta_ = beta
        self.n_iter_ = it + 1
        # standard errors from the inverse Hessian (Fisher information)
        self.se_ = np.sqrt(np.diag(np.linalg.inv(H)))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        return sigmoid(Xs @ self.beta_)


class XGModel:
    def __init__(self, seed: int = 7, benchmark: str = "Oracle (true p)"):
        self.seed = seed
        self.benchmark = benchmark   # real data: StatsBomb's own xG; synthetic: the simulator's true p

    def fit(self, shots: dict[str, np.ndarray]):
        y = (shots["outcome"] == "goal").astype(float)
        pen = shots["situation"] == "penalty"
        self.penalty_xg = float(y[pen].mean()) if pen.any() else 0.76
        X = featurize(shots["x"][~pen], shots["y"][~pen], shots["header"][~pen], shots["situation"][~pen])
        yy, true_p = y[~pen], shots["true_xg"][~pen]
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(yy))
        cut = int(0.8 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        self.model = LogisticRegressionIRLS(l2=1.0).fit(X[tr], yy[tr])
        p_te = self.model.predict_proba(X[te])
        base = np.full_like(p_te, yy[tr].mean())

        # "Distance-only" baseline - shows the value of the extra features.
        dist_only = LogisticRegressionIRLS(l2=1.0).fit(X[tr][:, [0]], yy[tr])
        p_dist = dist_only.predict_proba(X[te][:, [0]])

        z = self.model.beta_[1:] / self.model.se_[1:]
        bm = ~np.isnan(true_p[te].astype(float))
        tp, yb, pb = true_p[te][bm].astype(float), yy[te][bm], p_te[bm]
        self.report = {
            "n_shots": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
            "goal_rate": float(y.mean()), "penalty_xg": self.penalty_xg,
            "iterations": self.model.n_iter_,
            "convergence": [round(v, 6) for v in self.model.history_],
            "metrics": {
                "model": {"log_loss": log_loss(yy[te], p_te), "brier": float(np.mean((p_te - yy[te]) ** 2)),
                          "auc": roc_auc(yy[te], p_te)},
                "distance_only": {"log_loss": log_loss(yy[te], p_dist),
                                  "brier": float(np.mean((p_dist - yy[te]) ** 2)), "auc": roc_auc(yy[te], p_dist)},
                "base_rate": {"log_loss": log_loss(yy[te], base), "brier": float(np.mean((base - yy[te]) ** 2)),
                              "auc": 0.5},
                "oracle": {"log_loss": log_loss(yb, tp), "brier": float(np.mean((tp - yb) ** 2)), "auc": roc_auc(yb, tp)},
            },
            "corr_with_truth": float(np.corrcoef(pb, tp)[0, 1]),
            "benchmark_name": self.benchmark,
            "coefficients": [
                {"feature": f, "beta": float(b), "se": float(s), "z": float(zz),
                 "odds_ratio_per_sd": float(math.exp(b))}
                for f, b, s, zz in zip(FEATURES, self.model.beta_[1:], self.model.se_[1:], z)
            ],
            "intercept": float(self.model.beta_[0]),
            "calibration": calibration(yy[te], p_te),
            "roc": roc_curve(yy[te], p_te),
        }
        return self

    def predict(self, x, y, header, situation) -> np.ndarray:
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        header = np.atleast_1d(np.asarray(header, bool))
        situation = np.atleast_1d(np.asarray(situation, dtype=object))
        out = np.full(len(x), self.penalty_xg)
        m = situation != "penalty"
        if m.any():
            out[m] = self.model.predict_proba(featurize(x[m], y[m], header[m], situation[m]))
        return out

    def surface(self, header=False, situation="open_play", nx=40, ny=27):
        """xG evaluated over the attacking half - used for the contour map."""
        xs = np.linspace(60, 119.5, nx)
        ys = np.linspace(0.5, 79.5, ny)
        gx, gy = np.meshgrid(xs, ys)
        p = self.predict(gx.ravel(), gy.ravel(), np.full(gx.size, header), np.full(gx.size, situation))
        return {"x": xs.round(2).tolist(), "y": ys.round(2).tolist(),
                "z": p.reshape(ny, nx).round(4).tolist()}
