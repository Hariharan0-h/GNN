"""
Partition-parallel BSP executor (real execution, measured wall-clock time).

Workload
    S supersteps of GraphSAGE-style inference layers on F-dimensional vertex
    states:  h_v <- tanh( mean_{u in N(v)} h_u  @ W )   (h_v itself if deg(v) = 0).
    Only *active* vertices are recomputed, as in incremental GNN inference on
    dynamic graphs, so the work of a partition changes from window to window.
    The work of a row is proportional to deg(v) + F (aggregation + dense transform).

Execution model
    Every partition is a worker that owns its vertices and holds read-only ghost
    copies of remote neighbours. Partition i has a dedicated pool of tau_i threads
    (sum_i tau_i = T). A superstep is
        compute   each partition aggregates its active rows; rows are split into
                  tau_i chunks of equal adjacency work, executed by numba kernels
                  that release the GIL
        barrier
        exchange  owners publish the boundary values that changed (active boundary
                  vertices), every partition refreshes the corresponding ghosts
                  (cost proportional to the active communication volume)
    The superstep time is therefore governed by the slowest partition
    (W_i / tau_i) plus the halo exchange.

THREAD_ADJUST changes tau. REBALANCE changes the assignment (new layout).
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
from numba import njit

from graph.csr import CSRGraph


@njit(nogil=True, cache=True)
def _sage_layer(indptr, cols, X, out, rows, start, end, Wt):
    F = X.shape[1]
    acc = np.empty(F, dtype=np.float32)
    for i in range(start, end):
        r = rows[i]
        a = indptr[r]
        b = indptr[r + 1]
        if b == a:
            for f in range(F):
                acc[f] = X[r, f]
        else:
            for f in range(F):
                acc[f] = 0.0
            for j in range(a, b):
                c = cols[j]
                for f in range(F):
                    acc[f] += X[c, f]
            inv = np.float32(1.0 / (b - a))
            for f in range(F):
                acc[f] *= inv
        # dense transform: out[r] = tanh(acc @ W)   (Wt = W transposed, row-major)
        for o in range(F):
            s = np.float32(0.0)
            for f in range(F):
                s += acc[f] * Wt[o, f]
            out[r, o] = np.tanh(s)


@dataclass
class PartitionLayout:
    local: np.ndarray        # global ids of the owned vertices (L)
    indptr: np.ndarray       # CSR over owned rows
    cols: np.ndarray         # local column ids: 0..L-1 owned, L..L+G-1 ghosts
    ghosts: np.ndarray       # global ids of ghost vertices (G)
    send_local: np.ndarray   # local ids of owned vertices that are ghosts elsewhere
    send_global: np.ndarray  # their global ids


def build_layout(g: CSRGraph, assign: np.ndarray, k: int) -> list:
    n = g.n
    src = g.sources()
    dst = g.indices
    cross = assign[src] != assign[dst]
    is_sent = np.zeros(n, dtype=bool)
    is_sent[dst[cross]] = True
    pos = np.empty(n, dtype=np.int64)
    layouts = []
    for i in range(k):
        local = np.flatnonzero(assign == i).astype(np.int64)
        L = local.shape[0]
        starts = g.indptr[local]
        lens = g.indptr[local + 1] - starts
        indptr = np.zeros(L + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        idx = np.repeat(starts - indptr[:-1], lens) + np.arange(indptr[-1], dtype=np.int64)
        cols_g = g.indices[idx]
        ghosts = np.unique(cols_g[assign[cols_g] != i])
        pos[local] = np.arange(L)
        pos[ghosts] = L + np.arange(ghosts.shape[0])
        cols = pos[cols_g]
        send_mask = is_sent[local]
        layouts.append(PartitionLayout(
            local=local, indptr=indptr, cols=cols, ghosts=ghosts,
            send_local=np.flatnonzero(send_mask).astype(np.int64),
            send_global=local[send_mask],
        ))
    return layouts


def split_by_work(lens: np.ndarray, rows: np.ndarray, parts: int, row_cost: float = 1.0) -> np.ndarray:
    """Chunk boundaries (indices into rows) giving `parts` chunks of ~equal work (deg + row_cost)."""
    if rows.shape[0] == 0:
        return np.zeros(parts + 1, dtype=np.int64)
    work = np.cumsum(lens[rows] + row_cost)
    targets = work[-1] * np.arange(1, parts) / parts
    cuts = np.searchsorted(work, targets, side="left") + 1
    return np.concatenate([[0], np.minimum(cuts, rows.shape[0]), [rows.shape[0]]]).astype(np.int64)


def allocate_threads(work: np.ndarray, total: int) -> np.ndarray:
    """THREAD_ADJUST policy: threads proportional to work, at least 1 each, summing to `total`."""
    k = work.shape[0]
    if total < k:
        raise ValueError("need at least one thread per partition")
    tau = np.ones(k, dtype=np.int64)
    spare = total - k
    if spare == 0 or work.sum() <= 0:
        return even_threads(k, total)
    share = work / work.sum() * total - 1.0
    share = np.maximum(share, 0.0)
    if share.sum() > 0:
        share = share / share.sum() * spare
    extra = np.floor(share).astype(np.int64)
    rest = spare - extra.sum()
    order = np.argsort(-(share - extra), kind="mergesort")
    extra[order[:rest]] += 1
    return tau + extra


def even_threads(k: int, total: int) -> np.ndarray:
    tau = np.full(k, total // k, dtype=np.int64)
    tau[: total % k] += 1
    return np.maximum(tau, 1)


class BSPExecutor:

    def __init__(self, total_threads: int = 8, feature_dim: int = 64, supersteps: int = 10, seed: int = 0):
        self.total_threads = total_threads
        self.feature_dim = feature_dim
        self.supersteps = supersteps
        self.seed = seed
        self._pools = []
        self._tau = None
        self._x0 = None

    # ---------------- thread pools ------------------------------------------------------

    def configure_threads(self, tau) -> float:
        """(Re)create one pool per partition. Returns the reconfiguration time in seconds."""
        tau = [int(t) for t in tau]
        if self._tau == tau:
            return 0.0
        start = time.perf_counter()
        for pool in self._pools:
            pool.shutdown(wait=True)
        self._pools = [ThreadPoolExecutor(max_workers=t) for t in tau]
        # start the worker threads now so their creation is not timed as compute
        for pool, t in zip(self._pools, tau):
            wait([pool.submit(time.sleep, 0) for _ in range(t)])
        self._tau = tau
        return time.perf_counter() - start

    def close(self):
        for pool in self._pools:
            pool.shutdown(wait=True)
        self._pools = []
        self._tau = None

    # ---------------- state ---------------------------------------------------------------

    def initial_state(self, n: int) -> np.ndarray:
        if self._x0 is None or self._x0.shape[0] < n:
            rng = np.random.default_rng(self.seed)
            self._x0 = rng.standard_normal((max(n, 1), self.feature_dim)).astype(np.float32)
        return self._x0[:n]

    @property
    def weight(self) -> np.ndarray:
        """Layer weight W (F x F), fixed by the seed."""
        if getattr(self, "_w", None) is None:
            rng = np.random.default_rng(self.seed + 1)
            F = self.feature_dim
            self._w = (rng.standard_normal((F, F)) / np.sqrt(F)).astype(np.float32)
        return self._w

    def _buffer(self, slot: int, rows: int, F: int) -> np.ndarray:
        """A reusable float32 [rows, F] view; grows (with headroom) only when too small."""
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        buf = self._buffers.get(slot)
        if buf is None or buf.shape[0] < rows or buf.shape[1] != F:
            buf = np.empty((int(rows * 1.25) + 16, F), dtype=np.float32)
            buf[:] = 0.0  # touch the pages now, outside any timed region
            self._buffers[slot] = buf
        return buf[:rows]

    @staticmethod
    def row_work(degree: np.ndarray, feature_dim: int) -> np.ndarray:
        """Work of recomputing each vertex, in units of F-vector operations: deg(v) + F."""
        return degree.astype(np.float64) + float(feature_dim)

    # ---------------- run ----------------------------------------------------------------

    def run(self, g: CSRGraph, assign: np.ndarray, k: int, active: np.ndarray, tau,
            layout=None, X=None, return_state: bool = False) -> dict:
        t0 = time.perf_counter()
        if layout is None:
            layout = build_layout(g, assign, k)
        t_layout = time.perf_counter() - t0
        t_reconf = self.configure_threads(tau)

        X = self.initial_state(g.n) if X is None else X
        F = X.shape[1]
        Wt = np.ascontiguousarray(self.weight.T)
        cur, nxt, rows, bounds = [], [], [], []
        send_l, send_g, ghost_slot, ghost_g = [], [], [], []
        work = np.zeros(k, dtype=np.float64)
        for i, lay in enumerate(layout):
            # reuse preallocated buffers: fresh multi-MB allocations per run cause page
            # faults (and, under memory pressure, multi-second stalls) inside the timed region
            idx = np.concatenate([lay.local, lay.ghosts])
            a = self._buffer(2 * i, idx.shape[0], F)
            b = self._buffer(2 * i + 1, idx.shape[0], F)
            np.take(X, idx, axis=0, out=a)
            b[:] = a
            cur.append(a)
            nxt.append(b)
            r = np.flatnonzero(active[lay.local]).astype(np.int64)
            rows.append(r)
            lens = np.diff(lay.indptr)
            work[i] = float((lens[r] + F).sum())
            bounds.append(split_by_work(lens, r, int(tau[i]), row_cost=F))
            # incremental halo exchange: only values of active vertices change
            m_send = active[lay.send_global]
            send_l.append(lay.send_local[m_send])
            send_g.append(lay.send_global[m_send])
            g_idx = np.flatnonzero(active[lay.ghosts])
            ghost_slot.append(lay.local.shape[0] + g_idx)
            ghost_g.append(lay.ghosts[g_idx])
        Xg = self._buffer(-1, g.n, F)

        t_compute = 0.0
        t_exchange = 0.0
        for _ in range(self.supersteps):
            s0 = time.perf_counter()
            futures = []
            for i, lay in enumerate(layout):
                b = bounds[i]
                for c in range(b.shape[0] - 1):
                    if b[c + 1] > b[c]:
                        futures.append(self._pools[i].submit(
                            _sage_layer, lay.indptr, lay.cols, cur[i], nxt[i], rows[i], b[c], b[c + 1], Wt))
            wait(futures)
            s1 = time.perf_counter()
            for i in range(k):
                if send_l[i].shape[0]:
                    Xg[send_g[i]] = nxt[i][send_l[i]]
            for i in range(k):
                if ghost_slot[i].shape[0]:
                    nxt[i][ghost_slot[i]] = Xg[ghost_g[i]]
            s2 = time.perf_counter()
            cur, nxt = nxt, cur
            t_compute += s1 - s0
            t_exchange += s2 - s1

        result = {
            "time_exec": t_compute + t_exchange,
            "time_compute": t_compute,
            "time_exchange": t_exchange,
            "time_layout": t_layout,
            "time_reconfigure": t_reconf,
            "work": work,
            "ghosts": int(sum(l.ghosts.shape[0] for l in layout)),
            "active_ghosts": int(sum(s.shape[0] for s in ghost_slot)),
            "active_sent": int(sum(s.shape[0] for s in send_l)),
            "active": int(active.sum()),
        }
        if return_state:
            out = np.empty((g.n, F), dtype=np.float32)
            for i, lay in enumerate(layout):
                out[lay.local] = cur[i][: lay.local.shape[0]]
            result["state"] = out
        return result
