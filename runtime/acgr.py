"""
Adaptive Communication-aware Graph Repartitioning (ACGR)

ACGR refines an existing k-way partition by migrating boundary vertices.
It minimises the normalised objective

    C = alpha * cut / m
      + beta  * CV / (2m)
      + gamma * max(0, lambda - eps)
      + delta * min(skew, 10) / 10

  cut     edge cut
  CV      communication volume = number of ghost copies (graph.metrics.comm_volume).
          The previous version used "boundary edges", which is always 2 * cut and
          therefore duplicated the cut term.
  lambda  load imbalance max_i W_i / mean(W) - 1, where W_i is the (work-)weighted load.
          Imbalance inside the tolerance eps is a constraint, not a cost (as in METIS):
          penalising it would make every move away from a perfectly balanced partition
          look worse, which is what limited the original ACGR to a one-edge gain on Cora.
  skew    mean absolute degree skewness of the partitions

A move v: s -> t is admissible only if it respects the balance constraint
(paper Eq. 10):   W_t + w_v <= (1 + eps) * mean(W)   or   W_t + w_v < W_s
and it is accepted only if it strictly lowers C (paper Eq. 11).

Algorithm: best-first greedy boundary refinement. Every pass computes the best
admissible move of each boundary vertex, visits the vertices in order of
increasing objective change, re-evaluates each move against the current state
and applies it if it still improves C. Neighbour-partition counts are kept per
vertex, so evaluating a move costs O(deg(v) + k) and one pass costs O(|E| k).
"""

import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from graph import metrics as M
from graph.csr import CSRGraph, assign_to_partitions, partitions_to_assign


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _skew_of(n_p, s1, s2, s3):
    if n_p <= 1:
        return 0.0
    mu = s1 / n_p
    var = s2 / n_p - mu * mu
    if var <= 1e-12:
        return 0.0
    m3 = s3 / n_p - 3.0 * mu * s2 / n_p + 2.0 * mu * mu * mu
    return abs(m3 / var ** 1.5)


@njit(cache=True)
def _objective(cut, cv, W, nP, S1, S2, S3, k, m, alpha, beta, gamma, delta, eps):
    total = 0.0
    for p in range(k):
        total += W[p]
    mean = total / k
    lam = 0.0
    if mean > 0:
        mx = W[0]
        for p in range(1, k):
            if W[p] > mx:
                mx = W[p]
        lam = mx / mean - 1.0 - eps
        if lam < 0.0:
            lam = 0.0
    tot = 0.0
    cnt = 0
    for p in range(k):
        if nP[p] > 0:
            tot += _skew_of(nP[p], S1[p], S2[p], S3[p])
            cnt += 1
    sk = tot / cnt if cnt > 0 else 0.0
    if sk > 10.0:
        sk = 10.0
    mm = m if m > 0 else 1
    return alpha * cut / mm + beta * cv / (2.0 * mm) + gamma * lam + delta * sk / 10.0


@njit(cache=True)
def _best_move(v, indptr, indices, assign, cnt, k, w, deg, W, nP, S1, S2, S3,
               cut, cv, m, alpha, beta, gamma, delta, eps, cap):
    """Return (best target, new objective, new cut, new cv) or (-1, ...) if no admissible target."""
    s = assign[v]
    if nP[s] <= 1:
        return -1, 0.0, 0, 0
    d = float(deg[v])
    best_t = -1
    best_obj = 1e300
    best_cut = 0
    best_cv = 0
    for t in range(k):
        if t == s or cnt[v, t] == 0:
            continue
        if not (W[t] + w[v] <= cap or W[t] + w[v] < W[s]):
            continue
        new_cut = cut + cnt[v, s] - cnt[v, t]
        dcv = 0
        if cnt[v, s] > 0:
            dcv += 1
        if cnt[v, t] > 0:
            dcv -= 1
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            pu = assign[u]
            if pu != s and cnt[u, s] == 1:
                dcv -= 1
            if pu != t and cnt[u, t] == 0:
                dcv += 1
        new_cv = cv + dcv
        # temporarily apply the load / moment changes
        W[s] -= w[v]
        W[t] += w[v]
        nP[s] -= 1
        nP[t] += 1
        S1[s] -= d
        S1[t] += d
        S2[s] -= d * d
        S2[t] += d * d
        S3[s] -= d * d * d
        S3[t] += d * d * d
        obj = _objective(new_cut, new_cv, W, nP, S1, S2, S3, k, m, alpha, beta, gamma, delta, eps)
        W[s] += w[v]
        W[t] -= w[v]
        nP[s] += 1
        nP[t] -= 1
        S1[s] += d
        S1[t] -= d
        S2[s] += d * d
        S2[t] -= d * d
        S3[s] += d * d * d
        S3[t] -= d * d * d
        if obj < best_obj:
            best_obj = obj
            best_t = t
            best_cut = new_cut
            best_cv = new_cv
    return best_t, best_obj, best_cut, best_cv


@njit(cache=True)
def _acgr_core(indptr, indices, assign, k, w, deg, m, alpha, beta, gamma, delta,
               eps, max_passes, max_moves):
    n = indptr.shape[0] - 1
    cnt = np.zeros((n, k), dtype=np.int32)
    for v in range(n):
        for e in range(indptr[v], indptr[v + 1]):
            cnt[v, assign[indices[e]]] += 1

    W = np.zeros(k, dtype=np.float64)
    nP = np.zeros(k, dtype=np.float64)
    S1 = np.zeros(k, dtype=np.float64)
    S2 = np.zeros(k, dtype=np.float64)
    S3 = np.zeros(k, dtype=np.float64)
    cut2 = 0
    cv = 0
    for v in range(n):
        s = assign[v]
        d = float(deg[v])
        W[s] += w[v]
        nP[s] += 1.0
        S1[s] += d
        S2[s] += d * d
        S3[s] += d * d * d
        for p in range(k):
            if p != s and cnt[v, p] > 0:
                cv += 1
                cut2 += cnt[v, p]
    cut = cut2 // 2

    obj = _objective(cut, cv, W, nP, S1, S2, S3, k, m, alpha, beta, gamma, delta, eps)
    cap = (1.0 + eps) * W.sum() / k

    boundary = np.empty(n, dtype=np.int64)
    deltas = np.empty(n, dtype=np.float64)
    moves = 0
    passes = 0
    for _ in range(max_passes):
        passes += 1
        nb = 0
        for v in range(n):
            s = assign[v]
            is_boundary = False
            for p in range(k):
                if p != s and cnt[v, p] > 0:
                    is_boundary = True
                    break
            if not is_boundary:
                continue
            t, new_obj, _c, _v = _best_move(v, indptr, indices, assign, cnt, k, w, deg, W, nP,
                                            S1, S2, S3, cut, cv, m, alpha, beta, gamma, delta, eps, cap)
            if t >= 0 and new_obj < obj - 1e-12:
                boundary[nb] = v
                deltas[nb] = new_obj - obj
                nb += 1
        if nb == 0:
            break
        order = np.argsort(deltas[:nb], kind="mergesort")
        pass_moves = 0
        for i in range(nb):
            v = boundary[order[i]]
            t, new_obj, new_cut, new_cv = _best_move(v, indptr, indices, assign, cnt, k, w, deg, W,
                                                     nP, S1, S2, S3, cut, cv, m, alpha, beta,
                                                     gamma, delta, eps, cap)
            if t < 0 or new_obj >= obj - 1e-12:
                continue
            s = assign[v]
            d = float(deg[v])
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                cnt[u, s] -= 1
                cnt[u, t] += 1
            assign[v] = t
            W[s] -= w[v]
            W[t] += w[v]
            nP[s] -= 1.0
            nP[t] += 1.0
            S1[s] -= d
            S1[t] += d
            S2[s] -= d * d
            S2[t] += d * d
            S3[s] -= d * d * d
            S3[t] += d * d * d
            cut = new_cut
            cv = new_cv
            obj = new_obj
            moves += 1
            pass_moves += 1
            if moves >= max_moves:
                break
        if pass_moves == 0 or moves >= max_moves:
            break
    return obj, cut, cv, moves, passes


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

@dataclass
class ACGRResult:
    assign: np.ndarray
    objective_before: float
    objective_after: float
    cut_before: int
    cut_after: int
    cv_before: int
    cv_after: int
    imbalance_before: float
    imbalance_after: float
    moves: int
    migrated: int
    passes: int
    time_sec: float


class ACGR:

    def __init__(
        self,
        num_partitions=4,
        alpha=0.45,
        beta=0.35,
        gamma=0.15,
        delta=0.05,
        max_iterations=10,
        epsilon=0.03,
        max_move_fraction=1.0,
    ):
        self.num_partitions = num_partitions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.max_move_fraction = max_move_fraction

    # ---------------- balance tolerance -------------------------------------------------

    def effective_epsilon(self, w: np.ndarray) -> float:
        """
        Tolerance actually used for the constraint and the penalty. It always allows
        at least one (heaviest) vertex of slack; otherwise on small graphs a 3%
        tolerance is smaller than one vertex and no move is ever admissible.
        """
        mean = float(w.sum()) / self.num_partitions
        if mean <= 0:
            return self.epsilon
        cap = max((1.0 + self.epsilon) * mean, mean + float(w.max()))
        return cap / mean - 1.0

    # ---------------- objective (reference implementation, full recompute) -------------

    def objective(self, g: CSRGraph, assign: np.ndarray, weights=None) -> float:
        k = self.num_partitions
        m = max(g.m, 1)
        w = np.ones(g.n) if weights is None else np.asarray(weights, dtype=np.float64)
        lam = max(0.0, M.imbalance_max(M.loads(assign, k, w)) - self.effective_epsilon(w))
        skew = min(M.degree_skew(g, assign, k), 10.0) / 10.0
        return float(
            self.alpha * M.edge_cut(g, assign) / m
            + self.beta * M.comm_volume(g, assign, k) / (2.0 * m)
            + self.gamma * lam
            + self.delta * skew
        )

    # ---------------- refinement on CSR ------------------------------------------------

    def refine_csr(self, g: CSRGraph, assign: np.ndarray, weights=None) -> ACGRResult:
        k = self.num_partitions
        assign = np.ascontiguousarray(assign, dtype=np.int64)
        if assign.shape[0] != g.n or assign.min() < 0 or assign.max() >= k:
            raise ValueError("assignment does not match the graph / number of partitions")
        w = np.ones(g.n, dtype=np.float64) if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
        work = assign.copy()
        before = M.summary(g, assign, k, w)
        obj_before = self.objective(g, assign, w)
        max_moves = max(1, int(self.max_move_fraction * g.n))
        start = time.perf_counter()
        obj, cut, cv, moves, passes = _acgr_core(
            g.indptr, g.indices, work, k, w, g.degree.astype(np.int64), g.m,
            self.alpha, self.beta, self.gamma, self.delta,
            self.effective_epsilon(w), self.max_iterations, max_moves,
        )
        elapsed = time.perf_counter() - start
        lam_after = M.imbalance_max(M.loads(work, k, w))
        return ACGRResult(
            assign=work,
            objective_before=obj_before,
            objective_after=float(obj),
            cut_before=before["edge_cut"],
            cut_after=int(cut),
            cv_before=before["comm_volume"],
            cv_after=int(cv),
            imbalance_before=before["work_imbalance_max"],
            imbalance_after=lam_after,
            moves=int(moves),
            migrated=int((work != assign).sum()),
            passes=int(passes),
            time_sec=elapsed,
        )

    # ---------------- legacy networkx API (kept for the original scripts) --------------

    def refine(self, graph, partitions):
        """networkx graph + list of sets -> (list of sets, objective)."""
        g, nodes = CSRGraph.from_networkx(graph)
        partitions = [set(p) for p in partitions]
        if len(partitions) != self.num_partitions:
            raise ValueError(
                f"ACGR received {len(partitions)} partitions, expected {self.num_partitions}."
            )
        assign = partitions_to_assign(partitions, g.n, nodes)
        result = self.refine_csr(g, assign)
        return assign_to_partitions(result.assign, self.num_partitions, nodes), result.objective_after

    def cost(self, graph, partitions):
        g, nodes = CSRGraph.from_networkx(graph)
        assign = partitions_to_assign(partitions, g.n, nodes)
        return self.objective(g, assign)

    def edge_cut(self, graph, partitions):
        g, nodes = CSRGraph.from_networkx(graph)
        return M.edge_cut(g, partitions_to_assign(partitions, g.n, nodes))

    def comm_volume(self, graph, partitions):
        g, nodes = CSRGraph.from_networkx(graph)
        return M.comm_volume(g, partitions_to_assign(partitions, g.n, nodes), self.num_partitions)

    def boundary_cost(self, graph, partitions):
        """Legacy metric: directed boundary adjacencies. Always equals 2 * edge_cut."""
        return 2 * self.edge_cut(graph, partitions)

    def imbalance(self, partitions):
        return M.imbalance_std(np.array([len(p) for p in partitions], dtype=np.float64))

    def degree_skew(self, graph, partitions):
        g, nodes = CSRGraph.from_networkx(graph)
        return M.degree_skew(g, partitions_to_assign(partitions, g.n, nodes), self.num_partitions)
