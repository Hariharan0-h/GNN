"""
Graph partitioners used as initial partitions and as baselines.

All functions take a CSRGraph and return an int64 assignment array
(assign[v] = partition id of vertex v).

  metis            multilevel k-way partitioning (Karypis & Kumar 1998), via pymetis
  kl_recursive     recursive Kernighan-Lin bisection (the original GNN-AdaptGraph partitioner)
  hash_partition   random balanced assignment
  ldg              Linear Deterministic Greedy streaming partitioner (Stanton & Kliot 2012)
  fennel           FENNEL streaming partitioner (Tsourakakis et al. 2014)
  lp_refine        size-constrained label-propagation refinement (Spinner-style baseline)
  ldg_place        LDG placement of newly arrived vertices (used by the dynamic streams)
"""

import numpy as np
from numba import njit

from graph.csr import CSRGraph


# ---------------------------------------------------------------------------
# METIS
# ---------------------------------------------------------------------------

def _int_weights(weights, n):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    mean = w.mean() if w.size else 1.0
    if mean <= 0:
        return None
    return np.maximum(1, np.round(w / mean * 100.0)).astype(np.int64).tolist()


def metis(g: CSRGraph, k: int, seed: int = 0, weights=None, ufactor: int = 30) -> np.ndarray:
    import pymetis

    if k == 1:
        return np.zeros(g.n, dtype=np.int64)
    options = pymetis.Options(seed=int(seed), ufactor=int(ufactor))
    _, parts = pymetis.part_graph(
        k,
        pymetis.CSRAdjacency(g.indptr.tolist(), g.indices.tolist()),
        vweights=_int_weights(weights, g.n),
        options=options,
    )
    return np.asarray(parts, dtype=np.int64)


# ---------------------------------------------------------------------------
# Recursive Kernighan-Lin (networkx)
# ---------------------------------------------------------------------------

def kl_recursive(g: CSRGraph, k: int, seed: int = 0) -> np.ndarray:
    from networkx.algorithms.community import kernighan_lin_bisection

    G = g.to_networkx()
    parts = [set(range(g.n))]
    rng = np.random.default_rng(seed)
    while len(parts) < k:
        largest = max(parts, key=len)
        parts.remove(largest)
        if len(largest) <= 1:
            parts.append(largest)
            break
        p1, p2 = kernighan_lin_bisection(G.subgraph(largest), seed=int(rng.integers(1 << 31)))
        parts.extend([set(p1), set(p2)])
    assign = np.zeros(g.n, dtype=np.int64)
    for pid, part in enumerate(parts):
        assign[list(part)] = pid
    return assign


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

def hash_partition(g: CSRGraph, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assign = np.empty(g.n, dtype=np.int64)
    assign[rng.permutation(g.n)] = np.arange(g.n) % k
    return assign


# ---------------------------------------------------------------------------
# Streaming: LDG and FENNEL
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ldg_core(indptr, indices, order, assign, load, w, k, cap):
    counts = np.zeros(k, dtype=np.float64)
    for idx in range(order.shape[0]):
        v = order[idx]
        if assign[v] >= 0:
            continue
        counts[:] = 0.0
        for e in range(indptr[v], indptr[v + 1]):
            p = assign[indices[e]]
            if p >= 0:
                counts[p] += 1.0
        best = -1
        best_score = -1e300
        for p in range(k):
            if load[p] + w[v] > cap:
                continue
            score = counts[p] * (1.0 - load[p] / cap)
            if score > best_score or (score == best_score and load[p] < load[best]):
                best_score = score
                best = p
        if best < 0:  # every partition full: fall back to the least loaded one
            best = 0
            for p in range(1, k):
                if load[p] < load[best]:
                    best = p
        assign[v] = best
        load[best] += w[v]


@njit(cache=True)
def _fennel_core(indptr, indices, order, assign, load, w, k, cap, alpha, gamma):
    counts = np.zeros(k, dtype=np.float64)
    for idx in range(order.shape[0]):
        v = order[idx]
        if assign[v] >= 0:
            continue
        counts[:] = 0.0
        for e in range(indptr[v], indptr[v + 1]):
            p = assign[indices[e]]
            if p >= 0:
                counts[p] += 1.0
        best = -1
        best_score = -1e300
        for p in range(k):
            if load[p] + w[v] > cap:
                continue
            score = counts[p] - alpha * gamma * load[p] ** (gamma - 1.0)
            if score > best_score or (score == best_score and load[p] < load[best]):
                best_score = score
                best = p
        if best < 0:
            best = 0
            for p in range(1, k):
                if load[p] < load[best]:
                    best = p
        assign[v] = best
        load[best] += w[v]


def _stream_order(n, seed, order):
    if order is not None:
        return np.asarray(order, dtype=np.int64)
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


def _weights(n, weights):
    if weights is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def ldg(g: CSRGraph, k: int, seed: int = 0, order=None, weights=None, slack: float = 0.05) -> np.ndarray:
    w = _weights(g.n, weights)
    cap = (1.0 + slack) * w.sum() / k
    assign = np.full(g.n, -1, dtype=np.int64)
    load = np.zeros(k, dtype=np.float64)
    _ldg_core(g.indptr, g.indices, _stream_order(g.n, seed, order), assign, load, w, k, cap)
    return assign


def fennel(g: CSRGraph, k: int, seed: int = 0, order=None, weights=None,
           gamma: float = 1.5, nu: float = 1.1) -> np.ndarray:
    w = _weights(g.n, weights)
    total = w.sum()
    alpha = max(g.m, 1) * k ** (gamma - 1.0) / max(total, 1.0) ** gamma
    cap = nu * total / k
    assign = np.full(g.n, -1, dtype=np.int64)
    load = np.zeros(k, dtype=np.float64)
    _fennel_core(g.indptr, g.indices, _stream_order(g.n, seed, order), assign, load, w, k, cap, alpha, gamma)
    return assign


def ldg_place(g: CSRGraph, assign: np.ndarray, k: int, vertices, weights=None, slack: float = 0.05) -> np.ndarray:
    """Place the given (currently unassigned, assign == -1) vertices with LDG. Modifies assign in place."""
    w = _weights(g.n, weights)
    placed = assign >= 0
    load = np.bincount(assign[placed], weights=w[placed], minlength=k).astype(np.float64)
    cap = (1.0 + slack) * w.sum() / k
    _ldg_core(g.indptr, g.indices, np.asarray(vertices, dtype=np.int64), assign, load, w, k, cap)
    return assign


# ---------------------------------------------------------------------------
# Label-propagation refinement (baseline refiner)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _lp_core(indptr, indices, order, assign, load, w, k, cap, max_iters):
    counts = np.zeros(k, dtype=np.int64)
    total_moves = 0
    for _ in range(max_iters):
        moves = 0
        for idx in range(order.shape[0]):
            v = order[idx]
            s = assign[v]
            counts[:] = 0
            for e in range(indptr[v], indptr[v + 1]):
                counts[assign[indices[e]]] += 1
            best = s
            for p in range(k):
                if p == s:
                    continue
                if counts[p] > counts[best] and load[p] + w[v] <= cap:
                    best = p
            if best != s:
                assign[v] = best
                load[s] -= w[v]
                load[best] += w[v]
                moves += 1
        total_moves += moves
        if moves == 0:
            break
    return total_moves


def lp_refine(g: CSRGraph, assign: np.ndarray, k: int, seed: int = 0, weights=None,
              epsilon: float = 0.03, max_iters: int = 10) -> np.ndarray:
    w = _weights(g.n, weights)
    out = assign.copy()
    load = np.bincount(out, weights=w, minlength=k).astype(np.float64)
    cap = max((1.0 + epsilon) * w.sum() / k, load.max())
    order = np.random.default_rng(seed).permutation(g.n).astype(np.int64)
    _lp_core(g.indptr, g.indices, order, out, load, w, k, cap, max_iters)
    return out


PARTITIONERS = {
    "METIS": metis,
    "KL": kl_recursive,
    "LDG": ldg,
    "FENNEL": fennel,
    "Hash": hash_partition,
}
