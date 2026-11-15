"""
Player scouting engine - unsupervised learning written from scratch with NumPy.

  1. Feature engineering: per-90 rates (goals, xG, shots, passes, pass %, progressive
     passes, xT added, carries, dribble %, tackles, interceptions, box touches ...).
  2. Robust standardisation (z-scores).
  3. PCA via SVD -> 2-D "scouting map" + explained variance + loadings.
  4. k-means++ clustering -> data-driven player *roles*; k chosen by silhouette score.
  5. Cosine similarity in standardised space -> "find me a replacement for X".
"""
from __future__ import annotations

import numpy as np

FEATURES = [
    ("goals_p90", "Goals"), ("xg_p90", "xG"), ("shots_p90", "Shots"), ("xg_per_shot", "xG / shot"),
    ("passes_p90", "Passes"), ("pass_pct", "Pass %"), ("prog_passes_p90", "Progressive passes"),
    ("xt_p90", "xT added"), ("box_entries_p90", "Box entries"), ("carries_p90", "Carries"),
    ("dribble_pct", "Carry success %"), ("tackles_p90", "Tackles"), ("interceptions_p90", "Interceptions"),
    ("avg_x", "Avg. position (depth)"),
]
KEYS = [k for k, _ in FEATURES]


def pca(Z: np.ndarray, k: int = 2):
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = S ** 2 / (len(Z) - 1)
    return (U[:, :k] * S[:k]), Vt[:k], var[:k] / var.sum(), var / var.sum()


def kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 100):
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    inertia = float(((X - C[lab]) ** 2).sum())
    return lab, C, inertia


def silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


POS_GROUP = {"GK": "Goalkeeper", "CB": "Centre-back", "RB": "Full-back", "LB": "Full-back",
             "DM": "Midfielder", "CM": "Midfielder", "RW": "Winger", "LW": "Winger", "ST": "Striker"}
TRAIT_WORDS = {"xt_p90": "Creative", "prog_passes_p90": "Progressive", "tackles_p90": "Ball-winning",
               "interceptions_p90": "Reading-the-game", "xg_p90": "Goal-threat", "pass_pct": "Possession",
               "carries_p90": "Ball-carrying", "box_entries_p90": "Penetrative", "passes_p90": "Metronome"}


def name_cluster(members_pos: list[str], center_z: np.ndarray, group_mean_z: dict[str, np.ndarray]) -> str:
    """Label = defining trait (vs. players of the same position group) + dominant position group."""
    groups = [POS_GROUP[p] for p in members_pos]
    base = max(set(groups), key=groups.count)
    if base == "Goalkeeper":
        return base
    rel = center_z - group_mean_z[base]
    best = max(TRAIT_WORDS, key=lambda k: rel[KEYS.index(k)])
    return f"{TRAIT_WORDS[best]} {base}"


class ScoutingModel:
    def fit(self, rows: list[dict], seed: int = 11, k_range=range(5, 9)):
        self.rows = rows
        X = np.array([[r[k] for k in KEYS] for r in rows], dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.Z = Z
        coords, comps, evr, evr_all = pca(Z, 2)
        self.coords, self.components, self.evr, self.evr_all = coords, comps, evr, evr_all

        rng = np.random.default_rng(seed)
        best = None
        self.k_scores = []
        for k in k_range:
            runs = [kmeans_pp(Z, k, rng) for _ in range(5)]
            lab, C, inertia = min(runs, key=lambda r: r[2])
            sil = silhouette(Z, lab)
            self.k_scores.append({"k": k, "silhouette": sil, "inertia": inertia})
            if best is None or sil > best[0]:
                best = (sil, k, lab, C)
        _, self.k, self.labels, self.centers = best
        positions = np.array([r["position"] for r in rows])
        groups = np.array([POS_GROUP[p] for p in positions])
        group_mean = {g: Z[groups == g].mean(0) for g in set(groups)}
        names = []
        for j in range(self.k):
            nm = name_cluster(list(positions[self.labels == j]), self.centers[j], group_mean)
            while nm in names:
                nm += " (B)"
            names.append(nm)
        self.cluster_names = names
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        self.Zn = Z / norms
        self.pct = (np.argsort(np.argsort(X, axis=0), axis=0) / (len(X) - 1) * 100)
        self.index = {r["player_id"]: i for i, r in enumerate(rows)}
        return self

    def similar(self, player_id: int, n: int = 8):
        i = self.index[player_id]
        sims = self.Zn @ self.Zn[i]
        order = [j for j in np.argsort(-sims) if j != i][:n]
        return [{**self.summary(j), "similarity": float(sims[j])} for j in order]

    def summary(self, j: int):
        r = self.rows[j]
        return {"player_id": r["player_id"], "name": r["name"], "team": r["team"], "team_id": r["team_id"],
                "position": r["position"], "cluster": int(self.labels[j]),
                "role": self.cluster_names[self.labels[j]],
                "pc1": float(self.coords[j, 0]), "pc2": float(self.coords[j, 1]), "minutes": r["minutes"]}

    def profile(self, player_id: int):
        j = self.index[player_id]
        return {**self.summary(j), "stats": {k: float(self.rows[j][k]) for k in KEYS},
                "percentiles": [{"key": k, "label": lbl, "value": float(self.rows[j][k]),
                                 "percentile": float(self.pct[j, i])} for i, (k, lbl) in enumerate(FEATURES)]}

    def map(self):
        return {
            "players": [self.summary(j) for j in range(len(self.rows))],
            "clusters": [{"id": j, "name": self.cluster_names[j], "size": int((self.labels == j).sum())}
                         for j in range(self.k)],
            "explained_variance": self.evr.tolist(), "scree": self.evr_all.tolist(),
            "loadings": [{"feature": lbl, "pc1": float(self.components[0, i]), "pc2": float(self.components[1, i])}
                         for i, (_, lbl) in enumerate(FEATURES)],
            "k_selection": self.k_scores, "k": self.k,
        }
