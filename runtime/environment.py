"""
Adaptive runtime: replays a materialized dynamic-graph trace under a policy.

Per window t:
  1. ingest     apply the window's graph changes; new vertices are placed with LDG
  2. observe    build the controller state (8 features per partition + communication graph)
  3. decide     the policy picks an action
  4. apply      NO_ACTION       keep assignment and threads
                THREAD_ADJUST   tau_i proportional to the current work W_i (sum = T)
                REBALANCE       ACGR refinement balancing the current work, then threads
                                re-provisioned for the new partitioning
                REPARTITION     METIS from scratch, relabelled to maximise overlap
                                (scratch-remap), then threads re-provisioned
  5. execute    S supersteps on the BSPExecutor (measured) or the calibrated cost model (predicted)

Window cost = execution time + adaptation overhead, where the overhead is the
measured ACGR/METIS/controller time plus vertex migration.

Platform (`RuntimeConfig.network`)
  "sm"   shared memory: execution is the measured executor time; migration is charged
         at memory-copy bandwidth (10 GB/s)
  "10g"  emulated distributed deployment: measured compute and exchange, plus every halo
  "1g"   exchange charged at the network bandwidth with a per-message latency, and
         migration charged at the network bandwidth
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from graph import metrics as M
from graph import partitioners as P
from graph.csr import CSRGraph
from runtime.acgr import ACGR
from runtime.bsp_executor import BSPExecutor, allocate_threads, even_threads

NO_ACTION, THREAD_ADJUST, REBALANCE, REPARTITION = 0, 1, 2, 3
ACTION_NAMES = {0: "NO_ACTION", 1: "THREAD_ADJUST", 2: "REBALANCE", 3: "REPARTITION"}
NUM_FEATURES = 8
FEATURE_NAMES = [
    "load_share", "thread_pressure", "ghost_ratio", "cut_ratio",
    "cut_drift", "load_volatility", "size_share", "active_fraction",
]


@dataclass
class RuntimeConfig:
    k: int = 4
    total_threads: int = 12
    feature_dim: int = 64
    supersteps: int = 20
    network: str = "sm"
    acgr: dict = field(default_factory=lambda: dict(
        alpha=0.45, beta=0.35, gamma=0.15, delta=0.05, epsilon=0.05, max_iterations=3))
    seed: int = 0


NETWORKS = {
    # name: (bandwidth bytes/s, latency per message s); None = shared memory
    "sm": None,
    "10g": (1.25e9, 50e-6),
    "1g": (1.25e8, 100e-6),
}
MEMORY_BANDWIDTH = 1.0e10


@dataclass
class SystemState:
    assign: np.ndarray
    tau: np.ndarray
    ref_cut: np.ndarray       # per-partition cut ratio right after the last (re)partitioning
    prev_share: np.ndarray    # load share in the previous window

    def copy(self):
        return SystemState(self.assign.copy(), self.tau.copy(), self.ref_cut.copy(), self.prev_share.copy())


@dataclass
class Observation:
    x: np.ndarray             # [k, 8] partition features
    edge_index: np.ndarray    # [2, E] communication graph (both directions)
    edge_weight: np.ndarray   # [E] cut edges between the two partitions / m
    work: np.ndarray          # [k] current work per partition
    info: dict
    globals: np.ndarray = None  # [3] work scale, communication fraction, window time scale


@njit(cache=True)
def _monitor(indptr, indices, assign, active, k):
    """
    One O(|E| + |V|) pass computing every monitored quantity:
      cut_inc[p]  directed cut edges leaving partition p      inc[p]  directed edges of p
      mat[p, q]   directed edges from p to q (p != q)          ghosts[q] ghost copies held by q
      act_ghosts / act_sent / pairs: halo exchange of the active vertices
    """
    n = indptr.shape[0] - 1
    cut_inc = np.zeros(k, dtype=np.float64)
    inc = np.zeros(k, dtype=np.float64)
    mat = np.zeros((k, k), dtype=np.float64)
    ghosts = np.zeros(k, dtype=np.float64)
    pairs = np.zeros((k, k), dtype=np.bool_)
    seen = np.full(k, -1, dtype=np.int64)
    act_ghosts = 0
    act_sent = 0
    for v in range(n):
        pv = assign[v]
        sent = False
        for e in range(indptr[v], indptr[v + 1]):
            pu = assign[indices[e]]
            inc[pv] += 1.0
            if pu != pv:
                cut_inc[pv] += 1.0
                mat[pv, pu] += 1.0
                if seen[pu] != v:
                    seen[pu] = v
                    ghosts[pu] += 1.0
                    if active[v]:
                        act_ghosts += 1
                        pairs[pv, pu] = True
                        sent = True
        if sent:
            act_sent += 1
    return cut_inc, inc, mat, ghosts, act_ghosts, act_sent, int(pairs.sum())


def monitor(g: CSRGraph, assign: np.ndarray, k: int, active: np.ndarray) -> dict:
    cut_inc, inc, mat, ghosts, ag, asent, pairs = _monitor(
        g.indptr, g.indices, np.ascontiguousarray(assign, dtype=np.int64), np.ascontiguousarray(active), k)
    return dict(cut_inc=cut_inc, inc=inc, mat=mat, ghosts=ghosts, act_ghosts=ag, act_sent=asent, pairs=pairs)


def active_comm(g: CSRGraph, assign: np.ndarray, k: int, active: np.ndarray):
    """
    One halo exchange: (ghost copies of active vertices, active vertices that must be
    sent, number of (owner, receiver) partition pairs that exchange at least one value).
    """
    m = monitor(g, assign, k, active)
    return m["act_ghosts"], m["act_sent"], m["pairs"]


def cut_matrix(g: CSRGraph, assign: np.ndarray, k: int) -> np.ndarray:
    src = g.sources()
    a, b = assign[src], assign[g.indices]
    mask = a != b
    mat = np.zeros((k, k), dtype=np.float64)
    np.add.at(mat, (a[mask], b[mask]), 1.0)
    return mat  # each undirected cut edge counted in both (i,j) and (j,i)


def migration_bytes(g: CSRGraph, moved: np.ndarray, feature_dim: int) -> float:
    """Vertex state (F floats) + adjacency (8 bytes per edge) + id, per migrated vertex."""
    if moved.size == 0:
        return 0.0
    return float((4 * feature_dim + 8 * g.degree[moved] + 16).sum())


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class CostModel:
    """
    T_window = S * ( a + b * max_i W_i / s(tau_i) + c * X_comm )
      W_i     work of partition i (deg + F per active vertex)
      s(tau)  measured parallel speedup of tau threads
      X_comm  active ghosts + active sent vertices of one halo exchange
    Fitted by least squares on measured executor runs (experiments/calibrate.py).
    """

    def __init__(self, a=0.0, b=1e-6, c=1e-7, speedup=None, reconfigure=1e-3):
        self.a, self.b, self.c = a, b, c
        self.speedup = np.asarray(speedup if speedup is not None else [1.0] + list(range(1, 33)), dtype=np.float64)
        self.reconfigure = reconfigure

    def s(self, tau):
        tau = np.asarray(tau, dtype=np.int64)
        return self.speedup[np.minimum(tau, self.speedup.shape[0] - 1)]

    def terms(self, work, tau, comm):
        return float(np.max(work / self.s(tau))), float(comm)

    def predict(self, supersteps, work, tau, comm) -> float:
        x1, x2 = self.terms(work, tau, comm)
        return supersteps * (self.a + self.b * x1 + self.c * x2)

    def to_dict(self):
        return dict(a=self.a, b=self.b, c=self.c, speedup=self.speedup.tolist(), reconfigure=self.reconfigure)

    @classmethod
    def from_dict(cls, d):
        return cls(d["a"], d["b"], d["c"], d["speedup"], d.get("reconfigure", 1e-3))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class AdaptiveRuntime:

    def __init__(self, g0: CSRGraph, cfg: RuntimeConfig, cost_model: CostModel = None, executor: BSPExecutor = None):
        self.g0 = g0
        self.cfg = cfg
        self.cost_model = cost_model
        self.executor = executor
        self.acgr = ACGR(num_partitions=cfg.k, **cfg.acgr)

    # ---------------- helpers ----------------------------------------------------------

    def vertex_work(self, g: CSRGraph, active: np.ndarray) -> np.ndarray:
        return BSPExecutor.row_work(g.degree, self.cfg.feature_dim) * active

    def balance_weights(self, g, active):
        """Weights ACGR/METIS balance: current work, plus a small structural term for inactive vertices."""
        rw = BSPExecutor.row_work(g.degree, self.cfg.feature_dim)
        return rw * active + 1e-3 * rw

    def _cut_ratio(self, g, assign):
        k = self.cfg.k
        src = g.sources()
        pa = assign[src]
        cut = np.bincount(pa[pa != assign[g.indices]], minlength=k).astype(np.float64)
        inc = np.bincount(pa, minlength=k).astype(np.float64)
        return cut / np.maximum(inc, 1.0)

    # ---------------- lifecycle ---------------------------------------------------------

    def initial_state(self) -> SystemState:
        k = self.cfg.k
        assign = P.metis(self.g0, k, seed=self.cfg.seed)
        return SystemState(
            assign=assign,
            tau=even_threads(k, self.cfg.total_threads),
            ref_cut=self._cut_ratio(self.g0, assign),
            prev_share=np.ones(k),
        )

    def ingest(self, state: SystemState, w) -> SystemState:
        s = state.copy()
        n_new = w.g.n - s.assign.shape[0]
        if n_new > 0:
            s.assign = np.concatenate([s.assign, np.full(n_new, -1, dtype=np.int64)])
            P.ldg_place(w.g, s.assign, self.cfg.k, np.arange(w.g.n - n_new, w.g.n))
        return s

    def observe(self, state: SystemState, w) -> Observation:
        k = self.cfg.k
        g, assign, active = w.g, state.assign, w.active
        work = M.loads(assign, k, self.vertex_work(g, active))
        mean_work = max(work.mean(), 1e-9)
        share = work / mean_work
        per_thread = work / state.tau
        pressure = per_thread / max(per_thread.mean(), 1e-9)
        sizes = np.bincount(assign, minlength=k).astype(np.float64)
        mon = monitor(g, assign, k, active)
        ghosts = mon["ghosts"]
        cut_ratio = mon["cut_inc"] / np.maximum(mon["inc"], 1.0)
        act = np.bincount(assign[active], minlength=k).astype(np.float64)
        x = np.stack([
            share,
            pressure,
            ghosts / np.maximum(sizes, 1.0),
            cut_ratio,
            cut_ratio - state.ref_cut,
            np.abs(share - state.prev_share),
            sizes / max(g.n / k, 1.0),
            act / np.maximum(sizes, 1.0),
        ], axis=1).astype(np.float32)
        mat = mon["mat"]
        ii, jj = np.nonzero(mat)
        # global runtime features (monitored quantities): total work, share of the window
        # spent communicating, and the window time, for the current configuration
        ghosts_a, sent_a, pairs = mon["act_ghosts"], mon["act_sent"], mon["pairs"]
        t_window = self.cost_model.predict(self.cfg.supersteps, work, state.tau, ghosts_a + sent_a) \
            + self.network_time(ghosts_a, pairs)
        t_comm = self.cfg.supersteps * self.cost_model.c * (ghosts_a + sent_a) + self.network_time(ghosts_a, pairs)
        glob = np.array([
            np.log10(1.0 + work.sum()) / 7.0,
            t_comm / max(t_window, 1e-9),
            np.log10(1.0 + 1e3 * t_window) / 4.0,
        ], dtype=np.float32)
        info = {
            "edge_cut": int(mat.sum() // 2),
            "work_imbalance": float(work.max() / mean_work - 1.0),
            "pressure_max": float(pressure.max()),
            "cut_ratio": float(mat.sum() / 2 / max(g.m, 1)),
            "comm_fraction": float(glob[1]),
            "predicted_window_s": float(t_window),
        }
        return Observation(
            globals=glob,
            x=x, edge_index=np.stack([ii, jj]).astype(np.int64),
            edge_weight=(mat[ii, jj] / 2.0 / max(g.m, 1)).astype(np.float32),
            work=work, info=info,
        )

    def apply(self, state: SystemState, w, action: int):
        """Returns (new state, overhead dict). Overheads are measured or modelled seconds."""
        k = self.cfg.k
        s = state.copy()
        overhead = {"adapt": 0.0, "migration": 0.0, "migrated": 0}
        work = M.loads(s.assign, k, self.vertex_work(w.g, w.active))
        if action == THREAD_ADJUST:
            s.tau = allocate_threads(work, self.cfg.total_threads)
            overhead["adapt"] = self.cost_model.reconfigure if self.cost_model else 0.0
        elif action in (REBALANCE, REPARTITION):
            weights = self.balance_weights(w.g, w.active)
            start = time.perf_counter()
            if action == REBALANCE:
                new_assign = self.acgr.refine_csr(w.g, s.assign, weights).assign
            else:
                new_assign = relabel_to_overlap(P.metis(w.g, k, seed=self.cfg.seed, weights=weights), s.assign, k)
            overhead["adapt"] = time.perf_counter() - start
            moved = np.flatnonzero(new_assign != s.assign)
            overhead["migrated"] = int(moved.size)
            overhead["migration"] = migration_bytes(w.g, moved, self.cfg.feature_dim) / self.migration_bandwidth
            s.assign = new_assign
            # re-provision threads for the new partitioning (REBALANCE is a superset of THREAD_ADJUST)
            s.tau = allocate_threads(M.loads(s.assign, k, self.vertex_work(w.g, w.active)), self.cfg.total_threads)
            s.ref_cut = self._cut_ratio(w.g, s.assign)
        return s, overhead

    def end_window(self, state: SystemState, w) -> SystemState:
        s = state.copy()
        work = M.loads(s.assign, self.cfg.k, self.vertex_work(w.g, w.active))
        s.prev_share = work / max(work.mean(), 1e-9)
        return s

    # ---------------- execution ---------------------------------------------------------

    @property
    def migration_bandwidth(self) -> float:
        net = NETWORKS[self.cfg.network]
        return MEMORY_BANDWIDTH if net is None else net[0]

    def network_time(self, ghosts: int, pairs: int) -> float:
        """Emulated network cost of the halo exchanges of one window (0 in shared memory)."""
        net = NETWORKS[self.cfg.network]
        if net is None:
            return 0.0
        bandwidth, latency = net
        per_step = pairs * latency + ghosts * 4.0 * self.cfg.feature_dim / bandwidth
        return self.cfg.supersteps * per_step

    def predict(self, state: SystemState, w) -> float:
        k = self.cfg.k
        work = M.loads(state.assign, k, self.vertex_work(w.g, w.active))
        ghosts, sent, pairs = active_comm(w.g, state.assign, k, w.active)
        return (self.cost_model.predict(self.cfg.supersteps, work, state.tau, ghosts + sent)
                + self.network_time(ghosts, pairs))

    def network_time_of(self, assign: np.ndarray, w) -> float:
        ghosts, _, pairs = active_comm(w.g, assign, self.cfg.k, w.active)
        return self.network_time(ghosts, pairs)


def relabel_to_overlap(new: np.ndarray, old: np.ndarray, k: int) -> np.ndarray:
    """Permute the labels of `new` to maximise overlap with `old` (minimum migration)."""
    from scipy.optimize import linear_sum_assignment

    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (new, old), 1)
    rows, cols = linear_sum_assignment(-overlap)
    perm = np.empty(k, dtype=np.int64)
    perm[rows] = cols
    return perm[new]


CONTROLLER_ACTIONS = (NO_ACTION, THREAD_ADJUST, REBALANCE, REPARTITION)


def lookahead_costs(rt: AdaptiveRuntime, state: SystemState, trace, t: int, horizon: int,
                    actions=CONTROLLER_ACTIONS) -> dict:
    """
    Oracle cost of each action at window t: overhead + predicted execution of window t,
    then `horizon - 1` further windows with NO_ACTION (state after ingest of trace[t]).
    """
    costs = {}
    for a in actions:
        s, over = rt.apply(state, trace[t], a)
        total = over["adapt"] + over["migration"] + rt.predict(s, trace[t])
        s = rt.end_window(s, trace[t])
        for h in range(1, horizon):
            if t + h >= len(trace):
                break
            s = rt.ingest(s, trace[t + h])
            total += rt.predict(s, trace[t + h])
            s = rt.end_window(s, trace[t + h])
        costs[a] = total
    return costs
