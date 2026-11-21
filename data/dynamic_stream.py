"""
Dynamic graph streams.

A stream produces an initial graph and a sequence of windows. Each window adds and
removes edges, may add vertices, and marks the vertices that are active (touched)
in that window. The executor recomputes the active vertices and their neighbours.

TemporalStream    real timestamped interactions (SNAP sx-mathoverflow). Edges expire
                  after `lifetime_fraction` of the total time span without a new
                  interaction, so the topology both grows and shrinks.
SyntheticStream   controlled mutations on top of a base graph, driven by a Markov
                  chain over three regimes:
                    quiet    background activity only
                    hotspot  a large part of one community becomes active (transient load)
                    drift    vertices of community A rewire their edges to community B
                             (persistent structural change that degrades the partition)
                  plus vertex growth with preferential attachment.

materialize(stream) turns a stream into a list of WindowGraph objects (CSR graph,
active mask, new vertices). The graph sequence does not depend on the controller's
actions, so every policy (and the oracle's look-ahead) runs on the identical trace.
"""

from dataclasses import dataclass

import numpy as np

from graph.csr import CSRGraph

KEY = np.int64(1 << 31)


@dataclass
class Window:
    t: int
    n: int
    add: np.ndarray
    remove: np.ndarray
    touched: np.ndarray
    regime: str = ""


@dataclass
class WindowGraph:
    t: int
    g: CSRGraph
    active: np.ndarray
    new_vertices: np.ndarray
    regime: str


def _keys(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    keep = u != v
    return u[keep] * KEY + v[keep]


def _edges(keys) -> np.ndarray:
    keys = np.fromiter(keys, dtype=np.int64) if not isinstance(keys, np.ndarray) else keys
    return np.stack([keys // KEY, keys % KEY], axis=1) if keys.size else np.zeros((0, 2), dtype=np.int64)


def active_closure(g: CSRGraph, touched: np.ndarray) -> np.ndarray:
    """Touched vertices plus their 1-hop neighbours."""
    active = np.zeros(g.n, dtype=bool)
    touched = np.asarray(touched, dtype=np.int64)
    touched = touched[touched < g.n]
    if touched.size == 0:
        return active
    active[touched] = True
    starts = g.indptr[touched]
    lens = g.indptr[touched + 1] - starts
    total = int(lens.sum())
    if total:
        offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        active[g.indices[offs + np.arange(total)]] = True
    return active


def materialize(stream) -> list:
    n, edges = stream.initial()
    keys = set(_keys(edges).tolist())
    out = []
    for w in stream.windows():
        if w.remove.size:
            keys.difference_update(_keys(w.remove).tolist())
        if w.add.size:
            keys.update(_keys(w.add).tolist())
        g = CSRGraph.from_edges(w.n, _edges(np.fromiter(keys, dtype=np.int64, count=len(keys))))
        out.append(WindowGraph(
            t=w.t, g=g, active=active_closure(g, w.touched),
            new_vertices=np.arange(n, w.n, dtype=np.int64), regime=w.regime,
        ))
        n = w.n
    return out


def initial_graph(stream) -> CSRGraph:
    n, edges = stream.initial()
    return CSRGraph.from_edges(n, edges)


# ---------------------------------------------------------------------------
# Real temporal stream
# ---------------------------------------------------------------------------

class TemporalStream:
    """Windows over a timestamped interaction sequence (real: SNAP; or given arrays)."""

    def __init__(self, name="sx-mathoverflow", num_windows=30, warm_fraction=0.5, lifetime_fraction=0.25,
                 events=None, regime="temporal"):
        if events is None:
            from data.graphs import load_temporal
            self.src, self.dst, self.t, self.total_n = load_temporal(name)
        else:
            self.src, self.dst, self.t, self.total_n = events
        self.name = name
        self.regime = regime
        self.num_windows = num_windows
        self.warm = int(warm_fraction * self.t.shape[0])
        span = float(self.t[-1] - self.t[0])
        self.lifetime = None if lifetime_fraction is None else lifetime_fraction * span
        self._windows = None
        self._initial = None

    def _build(self):
        src, dst, t = self.src, self.dst, self.t
        last_seen = {}
        for a, b, ts in zip(src[: self.warm].tolist(), dst[: self.warm].tolist(), t[: self.warm].tolist()):
            key = (min(a, b), max(a, b))
            last_seen[key] = ts
        now = int(t[self.warm - 1])
        if self.lifetime is not None:
            last_seen = {k: v for k, v in last_seen.items() if v >= now - self.lifetime}
        n = int(max(src[: self.warm].max(), dst[: self.warm].max())) + 1
        self._initial = (n, np.array(list(last_seen.keys()), dtype=np.int64).reshape(-1, 2))

        present = set(last_seen.keys())
        bounds = np.linspace(self.warm, t.shape[0], self.num_windows + 1).astype(np.int64)
        windows = []
        for w in range(self.num_windows):
            lo, hi = int(bounds[w]), int(bounds[w + 1])
            s, d, ts_ = src[lo:hi].tolist(), dst[lo:hi].tolist(), t[lo:hi].tolist()
            add = []
            for a, b, ts in zip(s, d, ts_):
                key = (min(a, b), max(a, b))
                if key not in present:
                    present.add(key)
                    add.append(key)
                last_seen[key] = ts
            now = ts_[-1] if ts_ else now
            remove = []
            if self.lifetime is not None:
                limit = now - self.lifetime
                expired = [k for k, v in last_seen.items() if v < limit]
                for key in expired:
                    del last_seen[key]
                    present.discard(key)
                remove = expired
            n = max(n, int(max(s + d)) + 1) if s else n
            touched = np.unique(np.concatenate([src[lo:hi], dst[lo:hi]]))
            windows.append(Window(
                t=w, n=n,
                add=np.array(add, dtype=np.int64).reshape(-1, 2),
                remove=np.array(remove, dtype=np.int64).reshape(-1, 2),
                touched=touched, regime=self.regime,
            ))
        self._windows = windows

    def initial(self):
        if self._initial is None:
            self._build()
        return self._initial

    def windows(self):
        if self._windows is None:
            self._build()
        return self._windows


# ---------------------------------------------------------------------------
# Synthetic temporal interaction stream (training only)
# ---------------------------------------------------------------------------

class _Ring:
    """Fixed-size ring buffer with O(1) uniform sampling (recency x activity weighting)."""

    def __init__(self, size):
        self.buf = [0] * size
        self.size = size
        self.count = 0

    def push(self, x):
        self.buf[self.count % self.size] = x
        self.count += 1

    def sample(self, rng):
        return self.buf[int(rng.integers(min(self.count, self.size)))]

    def __len__(self):
        return min(self.count, self.size)


def generate_interactions(rng, num_events, p_new=0.05, topics=16, homophily=0.8, p_triadic=0.4,
                          buffer=4000, topic_drift=0.15, max_adj=64):
    """
    Generative model of a growing interaction graph (cf. temporal networks such as
    Q&A and messaging platforms):
      * a new vertex arrives with probability p_new and joins a topic drawn from a
        topic popularity that drifts over time (log-normal random walk);
      * otherwise the initiator is a recently active vertex (uniform over the last
        `buffer` endpoints = recency- and activity-weighted);
      * the partner is a friend-of-friend (triadic closure, p_triadic), a recently active
        vertex of the same topic (homophily), or any recently active vertex.
    Returns (src, dst, t, n) with vertex ids in order of first appearance.
    """
    src = np.empty(num_events, dtype=np.int64)
    dst = np.empty(num_events, dtype=np.int64)
    topic, adj = [], []
    recent = _Ring(buffer)
    recent_topic = [_Ring(max(64, buffer // topics)) for _ in range(topics)]
    log_pop = rng.normal(0, 1, topics)

    def new_vertex():
        pop = np.exp(log_pop - log_pop.max())
        topic.append(int(rng.choice(topics, p=pop / pop.sum())))
        adj.append([])
        return len(topic) - 1

    for _ in range(10):
        new_vertex()
    e = 0
    while e < num_events:
        if e % 500 == 0:
            log_pop += rng.normal(0, topic_drift, topics)
        u = new_vertex() if (rng.random() < p_new or len(recent) < 20) else recent.sample(rng)
        r = rng.random()
        if r < p_triadic and adj[u]:
            w = adj[u][int(rng.integers(len(adj[u])))]
            v = adj[w][int(rng.integers(len(adj[w])))] if adj[w] else w
        elif rng.random() < homophily and len(recent_topic[topic[u]]) > 0:
            v = recent_topic[topic[u]].sample(rng)
        elif len(recent) > 0:
            v = recent.sample(rng)
        else:
            v = new_vertex()
        if v == u:
            continue
        src[e], dst[e] = u, v
        for a, b in ((u, v), (v, u)):
            if len(adj[a]) >= max_adj:
                adj[a][int(rng.integers(max_adj))] = b
            else:
                adj[a].append(b)
            recent.push(a)
            recent_topic[topic[a]].push(a)
        e += 1
    # remap to order of first appearance (generation order already is, except unused ids)
    order = np.stack([src, dst], axis=1).ravel()
    _, first = np.unique(order, return_index=True)
    labels = order[np.sort(first)]
    remap = np.full(len(topic), -1, dtype=np.int64)
    remap[labels] = np.arange(labels.shape[0])
    return remap[src], remap[dst], np.arange(num_events, dtype=np.int64), int(labels.shape[0])


def synthetic_temporal_stream(rng, num_windows=16, **overrides):
    """A TemporalStream over generated interactions with randomised parameters."""
    params = dict(
        num_events=int(rng.integers(60_000, 250_000)),
        p_new=float(rng.uniform(0.02, 0.10)),
        topics=int(rng.integers(8, 33)),
        homophily=float(rng.uniform(0.5, 0.95)),
        p_triadic=float(rng.uniform(0.2, 0.6)),
        buffer=int(rng.integers(2000, 20000)),
        topic_drift=float(rng.uniform(0.05, 0.3)),
    )
    params.update(overrides)
    events = generate_interactions(rng, **params)
    stream = TemporalStream(name="synthetic-temporal", num_windows=num_windows,
                            warm_fraction=float(rng.uniform(0.3, 0.6)),
                            lifetime_fraction=float(rng.uniform(0.15, 0.4)),
                            events=events, regime="temporal")
    return stream, params


# ---------------------------------------------------------------------------
# Synthetic stream
# ---------------------------------------------------------------------------

REGIMES = ("quiet", "hotspot", "drift")


class SyntheticStream:

    def __init__(self, base: CSRGraph, num_windows=30, seed=0, communities=16, p_stay=0.75,
                 growth=0.003, drift_fraction=0.25, hot_fraction=0.6, background=0.01,
                 regime_weights=(1.0, 1.0, 1.0)):
        self.base = base
        self.num_windows = num_windows
        self.seed = seed
        self.communities = communities
        self.p_stay = p_stay
        self.growth = growth
        self.drift_fraction = drift_fraction
        self.hot_fraction = hot_fraction
        self.background = background
        self.regime_weights = np.asarray(regime_weights, dtype=np.float64)
        self._windows = None

    def initial(self):
        return self.base.n, self.base.edges()

    def windows(self):
        if self._windows is None:
            self._windows = self._simulate()
        return self._windows

    def _next_regime(self, rng, current):
        if current is not None and rng.random() < self.p_stay:
            return current
        weights = self.regime_weights.copy()
        if current is not None:
            weights[REGIMES.index(current)] = 0.0
        return REGIMES[int(rng.choice(3, p=weights / weights.sum()))]

    def _simulate(self):
        from graph.partitioners import metis

        rng = np.random.default_rng(self.seed)
        g = self.base
        n = g.n
        c = max(2, min(self.communities, n // 20))
        comm = list(metis(g, c, seed=self.seed).tolist())
        adj = [set(g.neighbors(v).tolist()) for v in range(n)]
        members = [set() for _ in range(c)]
        for v, cv in enumerate(comm):
            members[cv].add(v)
        avg_deg = max(1.0, 2.0 * g.m / max(n, 1))

        regime = None
        hot = drift_a = drift_b = None
        windows = []
        for t in range(self.num_windows):
            new_regime = self._next_regime(rng, regime if t > 0 else None) if t > 0 else "quiet"
            if new_regime != regime:
                hot = int(rng.integers(c))
                drift_a, drift_b = (int(x) for x in rng.choice(c, 2, replace=False))
            regime = new_regime
            add, remove, touched = [], [], []

            if regime == "drift" and members[drift_a]:
                pool = list(members[drift_a])
                movers = rng.choice(pool, size=max(1, int(self.drift_fraction * len(pool))), replace=False)
                targets_b = np.array(sorted(members[drift_b]), dtype=np.int64)
                for v in movers.tolist():
                    same = [u for u in adj[v] if comm[u] == drift_a]
                    drop = same[: max(1, len(same) // 2 + 1)] if same else []
                    for u in drop:
                        adj[v].discard(u)
                        adj[u].discard(v)
                        remove.append((v, u))
                    if targets_b.size:
                        p = np.array([len(adj[u]) + 1.0 for u in targets_b])
                        new_nb = rng.choice(targets_b, size=min(targets_b.size, max(1, len(drop))),
                                            replace=False, p=p / p.sum())
                        for u in new_nb.tolist():
                            if u != v and u not in adj[v]:
                                adj[v].add(u)
                                adj[u].add(v)
                                add.append((v, u))
                    members[drift_a].discard(v)
                    members[drift_b].add(v)
                    comm[v] = drift_b
                    touched.append(v)

            # growth: new vertices join a community by preferential attachment
            target_comm = {"drift": drift_b, "hotspot": hot}.get(regime)
            for _ in range(max(1, int(self.growth * n))):
                cv = target_comm if target_comm is not None else int(rng.integers(c))
                pool = np.array(sorted(members[cv]), dtype=np.int64)
                v = n
                n += 1
                adj.append(set())
                comm.append(cv)
                members[cv].add(v)
                if pool.size:
                    d = int(min(pool.size, max(1, rng.poisson(avg_deg))))
                    p = np.array([len(adj[u]) + 1.0 for u in pool])
                    for u in rng.choice(pool, size=d, replace=False, p=p / p.sum()).tolist():
                        adj[v].add(u)
                        adj[u].add(v)
                        add.append((v, u))
                touched.append(v)

            if regime == "hotspot" and members[hot]:
                pool = np.array(sorted(members[hot]), dtype=np.int64)
                touched.extend(rng.choice(pool, size=max(1, int(self.hot_fraction * pool.size)),
                                          replace=False).tolist())
            touched.extend(rng.choice(n, size=max(1, int(self.background * n)), replace=False).tolist())

            windows.append(Window(
                t=t, n=n,
                add=np.array(add, dtype=np.int64).reshape(-1, 2),
                remove=np.array(remove, dtype=np.int64).reshape(-1, 2),
                touched=np.unique(np.array(touched, dtype=np.int64)),
                regime=regime,
            ))
        return windows
