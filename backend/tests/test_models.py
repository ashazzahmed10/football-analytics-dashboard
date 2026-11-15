import math

import numpy as np

from app.analytics.network import betweenness, pagerank
from app.analytics.ratings import DixonColes, Elo, rps, simulate_season
from app.analytics.scouting import kmeans_pp, pca, silhouette
from app.analytics.xg import LogisticRegressionIRLS, roc_auc
from app.analytics.xt import ExpectedThreat
from app.data.engine import shot_geometry, true_goal_probability
from app.data.league import round_robin


def test_shot_geometry():
    d, a = shot_geometry(108, 40)
    assert math.isclose(d, 12)
    assert math.isclose(a, 2 * math.atan(4 / 12), rel_tol=1e-9)
    assert shot_geometry(119, 40)[1] > shot_geometry(100, 40)[1] > shot_geometry(100, 5)[1]


def test_true_probability_monotone_in_distance():
    ps = [true_goal_probability(*shot_geometry(x, 40), False, "open_play", 0, 0) for x in (115, 105, 95, 85)]
    assert ps == sorted(ps, reverse=True)


def test_round_robin_is_valid_double_round_robin():
    rounds = round_robin(20)
    assert len(rounds) == 38
    games = [g for r in rounds for g in r]
    assert len(set(games)) == 380                     # every ordered pair exactly once
    for r in rounds:
        teams = [t for g in r for t in g]
        assert len(set(teams)) == 20                  # each team plays once per matchday


def test_irls_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2))
    beta = np.array([-1.0, 1.5, -0.7])
    p = 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))
    y = (rng.random(len(p)) < p).astype(float)
    m = LogisticRegressionIRLS(l2=0.0).fit(X, y)
    # convert standardised coefficients back to raw scale
    raw = m.beta_[1:] / m.sd_
    assert np.allclose(raw, beta[1:], atol=0.08)
    assert m.n_iter_ < 15


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.3).astype(int)
    p = rng.random(300) + 0.3 * y
    pos, neg = p[y == 1], p[y == 0]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert math.isclose(roc_auc(y, p), brute, rel_tol=1e-9)


def test_dixon_coles_recovers_strengths():
    rng = np.random.default_rng(2)
    n = 10
    att, dfn = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    att -= att.mean(); dfn -= dfn.mean()
    H, A = [], []
    for _ in range(12):
        for i in range(n):
            for j in range(n):
                if i != j:
                    H.append(i); A.append(j)
    H, A = np.array(H), np.array(A)
    hg = rng.poisson(np.exp(0.2 + 0.25 + att[H] - dfn[A]))
    ag = rng.poisson(np.exp(0.2 + att[A] - dfn[H]))
    m = DixonColes(xi=0.0).fit(range(n), H, A, hg, ag, np.zeros(len(H)))
    assert np.corrcoef(m.att, att)[0, 1] > 0.9
    assert abs(m.home - 0.25) < 0.08
    probs = m.probs(0, 1)
    assert math.isclose(probs.sum(), 1, rel_tol=1e-6)


def test_rps_bounds():
    assert rps(np.array([1, 0, 0]), 0) == 0
    assert rps(np.array([0, 0, 1]), 0) == 1
    assert 0 < rps(np.full(3, 1 / 3), 1) < rps(np.full(3, 1 / 3), 0)


def test_monte_carlo_probabilities_are_distributions():
    rng = np.random.default_rng(3)
    n = 6
    H = np.repeat(np.arange(n), n); A = np.tile(np.arange(n), n); m = H != A
    H, A = H[m], A[m]
    model = DixonColes().fit(range(n), H, A, rng.poisson(1.4, len(H)), rng.poisson(1.1, len(H)), np.zeros(len(H)))
    res = simulate_season(model, list(range(n)), [], list(zip(H, A)), n_sims=2000)
    P = np.array([r["positions"] for r in res])
    assert np.allclose(P.sum(0), 1, atol=1e-3) and np.allclose(P.sum(1), 1, atol=1e-3)


def test_elo_is_zero_sum():
    ms = [{"id": i, "matchday": i, "home_id": i % 3, "away_id": (i + 1) % 3, "home_goals": i % 4, "away_goals": 1}
          for i in range(30)]
    e = Elo().run([0, 1, 2], ms)
    assert math.isclose(sum(e.ratings.values()), 1500 * 3, rel_tol=1e-9)


def test_pagerank_and_betweenness():
    # path graph 0 <-> 1 <-> 2 : node 1 is the only bridge
    W = np.array([[0, 5, 0], [5, 0, 5], [0, 5, 0]], float)
    pr, _ = pagerank(W)
    assert math.isclose(pr.sum(), 1) and pr[1] > pr[0]
    bc = betweenness(W)
    assert bc[1] == bc.max() and bc[0] == 0


def test_xt_value_increases_towards_goal():
    rng = np.random.default_rng(4)
    n = 40000
    x, y = rng.uniform(0, 120, n), rng.uniform(0, 80, n)
    ex, ey = np.clip(x + rng.normal(8, 10, n), 0, 119.9), np.clip(y + rng.normal(0, 10, n), 0, 79.9)
    sx, sy = rng.uniform(95, 120, 3000), rng.uniform(25, 55, 3000)
    goals = rng.random(3000) < (sx - 90) / 60
    xt = ExpectedThreat().fit({"x": x, "y": y, "end_x": ex, "end_y": ey, "success": rng.random(n) < 0.8},
                              {"x": sx, "y": sy, "goal": goals})
    assert 0 <= xt.grid.min() and xt.grid.max() < 1
    assert xt.value([115], [40])[0] > xt.value([60], [40])[0] > xt.value([10], [40])[0]


def test_pca_and_kmeans():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.3, (50, 4)) for c in (0, 3, 6)])
    _, _, evr, _ = pca(X, 2)
    assert evr[0] > 0.9
    lab, _, _ = kmeans_pp(X, 3, rng)
    assert len(set(lab[:50])) == 1 and silhouette(X, lab) > 0.7
