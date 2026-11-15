"""
Pass-network graph analytics (no networkx - algorithms written by hand).

Nodes are players, directed weighted edges are completed passes passer -> receiver.
  * PageRank           - power iteration on the column-stochastic pass matrix.
                         "Who does the ball flow through?"
  * Betweenness        - Brandes' algorithm with Dijkstra, edge length = 1 / passes.
                         "Who connects the team's units?"
  * Weighted degree, network density and centralisation (dependency on one player).
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np


def pagerank(W: np.ndarray, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200):
    n = W.shape[0]
    out = W.sum(1, keepdims=True)
    P = np.where(out > 0, W / np.where(out == 0, 1, out), 1.0 / n)   # row-stochastic, dangling -> uniform
    r = np.full(n, 1.0 / n)
    for it in range(max_iter):
        new = (1 - damping) / n + damping * (r @ P)
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r / r.sum(), it + 1


def betweenness(W: np.ndarray) -> np.ndarray:
    """Brandes (2001) for weighted directed graphs."""
    n = W.shape[0]
    adj = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0 and i != j:
                adj[i].append((j, 1.0 / W[i, j]))
    cb = np.zeros(n)
    for s in range(n):
        stack, pred = [], defaultdict(list)
        sigma = np.zeros(n); sigma[s] = 1
        dist = np.full(n, np.inf); dist[s] = 0
        pq = [(0.0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist[v]:
                continue
            stack.append(v)
            for w, length in adj[v]:
                nd = d + length
                if nd < dist[w] - 1e-12:
                    dist[w] = nd
                    heapq.heappush(pq, (nd, w))
                    sigma[w] = sigma[v]
                    pred[w] = [v]
                elif abs(nd - dist[w]) <= 1e-12:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        seen = set()
        for w in reversed(stack):
            if w in seen:
                continue
            seen.add(w)
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    norm = (n - 1) * (n - 2)
    return cb / norm if norm else cb


def build_pass_network(passes: list[dict], players: dict[int, dict], min_edge: int = 3):
    """passes: successful passes of one team in one match (player_id, receiver_id, x, y, end_x, end_y)."""
    ids = sorted({p["player_id"] for p in passes} | {p["receiver_id"] for p in passes if p["receiver_id"]})
    ix = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    W = np.zeros((n, n))
    loc = defaultdict(list)
    for p in passes:
        if p["receiver_id"] not in ix:
            continue
        W[ix[p["player_id"]], ix[p["receiver_id"]]] += 1
        loc[p["player_id"]].append((p["x"], p["y"]))
        loc[p["receiver_id"]].append((p["end_x"], p["end_y"]))
    pr, pr_iter = pagerank(W)
    bc = betweenness(W)
    deg = W.sum(0) + W.sum(1)
    density = float((W > 0).sum() / (n * (n - 1))) if n > 1 else 0.0
    # Freeman-style degree centralisation: 0 = evenly shared, 1 = everything via one player
    centralisation = float(((deg.max() - deg).sum()) / ((n - 1) * deg.max())) if n > 1 and deg.max() else 0.0

    nodes = []
    for pid, i in ix.items():
        xs = np.array(loc[pid]) if loc[pid] else np.array([[60.0, 40.0]])
        pl = players.get(pid, {})
        nodes.append({
            "id": pid, "name": pl.get("name", str(pid)), "position": pl.get("position", ""),
            "shirt": pl.get("shirt"), "x": float(xs[:, 0].mean()), "y": float(xs[:, 1].mean()),
            "touches": int(len(loc[pid])), "passes_made": int(W[i].sum()), "passes_received": int(W[:, i].sum()),
            "pagerank": float(pr[i]), "betweenness": float(bc[i]),
        })
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            c = W[a, b] + W[b, a]
            if c >= min_edge:
                edges.append({"source": ids[a], "target": ids[b], "count": int(c),
                              "a_to_b": int(W[a, b]), "b_to_a": int(W[b, a])})
    return {"nodes": nodes, "edges": edges,
            "metrics": {"density": density, "centralisation": centralisation, "pagerank_iterations": pr_iter,
                        "total_passes": int(W.sum())}}
