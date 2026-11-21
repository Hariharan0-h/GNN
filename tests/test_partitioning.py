import itertools

import networkx as nx
import numpy as np
import pytest

from graph import metrics as M
from graph import partitioners as P
from graph.csr import CSRGraph, assign_to_partitions, partitions_to_assign
from runtime.acgr import ACGR


def random_graphs(count=50, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(count):
        kind = i % 3
        n = int(rng.integers(60, 400))
        s = int(rng.integers(1 << 30))
        if kind == 0:
            G = nx.planted_partition_graph(4, n // 4, 0.15, 0.01, seed=s)
        elif kind == 1:
            G = nx.barabasi_albert_graph(n, int(rng.integers(1, 5)), seed=s)
        else:
            G = nx.erdos_renyi_graph(n, 0.03, seed=s)
        yield CSRGraph.from_networkx(nx.convert_node_labels_to_integers(G))[0], rng


def brute_comm_volume(g, assign):
    total = 0
    for v in range(g.n):
        remote = {int(assign[u]) for u in g.neighbors(v)} - {int(assign[v])}
        total += len(remote)
    return total


def test_csr_roundtrip():
    G = nx.karate_club_graph()
    g, nodes = CSRGraph.from_networkx(G)
    assert g.n == G.number_of_nodes()
    assert g.m == G.number_of_edges()
    assert sorted(map(tuple, g.edges().tolist())) == sorted(tuple(sorted(e)) for e in G.edges())


def test_metrics_match_bruteforce():
    for g, rng in itertools.islice(random_graphs(), 15):
        k = int(rng.integers(2, 9))
        assign = rng.integers(0, k, g.n)
        G = g.to_networkx()
        parts = assign_to_partitions(assign, k)
        brute_cut = sum(1 for u, v in G.edges() if assign[u] != assign[v])
        assert M.edge_cut(g, assign) == brute_cut
        assert M.comm_volume(g, assign, k) == brute_comm_volume(g, assign)
        assert M.ghosts_per_partition(g, assign, k).sum() == M.comm_volume(g, assign, k)
        assert partitions_to_assign(parts, g.n).tolist() == assign.tolist()


@pytest.mark.parametrize("name", ["METIS", "LDG", "FENNEL", "Hash", "KL"])
def test_partitioners_are_valid_covers(name):
    for g, rng in itertools.islice(random_graphs(seed=1), 6):
        k = 4
        assign = P.PARTITIONERS[name](g, k, seed=3)
        assert assign.shape == (g.n,)
        assert assign.min() >= 0 and assign.max() < k
        sizes = np.bincount(assign, minlength=k)
        assert sizes.sum() == g.n
        if name in ("METIS", "LDG", "FENNEL", "Hash"):
            assert sizes.max() <= 1.15 * g.n / k + 2


def test_ldg_place_only_touches_unassigned():
    g, _ = next(random_graphs(seed=2))
    assign = P.metis(g, 4, seed=0)
    new = np.arange(g.n - 20, g.n)
    assign[new] = -1
    before = assign.copy()
    P.ldg_place(g, assign, 4, new)
    assert (assign >= 0).all()
    assert (assign[: g.n - 20] == before[: g.n - 20]).all()


def test_acgr_incremental_matches_full_recompute():
    """Exit check of Day 1: incremental gains == full recompute on 50 random graphs."""
    for g, rng in random_graphs(count=50, seed=7):
        k = int(rng.choice([2, 4, 8]))
        start = rng.choice(["Hash", "LDG"])
        assign = P.PARTITIONERS[start](g, k, seed=int(rng.integers(1000)))
        weights = rng.uniform(0.5, 3.0, g.n) if rng.random() < 0.5 else None
        acgr = ACGR(num_partitions=k, epsilon=0.05, max_iterations=8)
        res = acgr.refine_csr(g, assign, weights)

        w = np.ones(g.n) if weights is None else weights
        assert res.cut_after == M.edge_cut(g, res.assign)
        assert res.cv_after == M.comm_volume(g, res.assign, k)
        assert res.objective_after == pytest.approx(acgr.objective(g, res.assign, w), abs=1e-9)
        assert res.objective_after <= res.objective_before + 1e-12
        assert res.migrated == int((res.assign != assign).sum())

        # balance constraint (Eq. 10): the maximum load never exceeds max(cap, initial max)
        load0 = M.loads(assign, k, w)
        load1 = M.loads(res.assign, k, w)
        cap = (1 + acgr.effective_epsilon(w)) * w.sum() / k
        assert load1.max() <= max(cap, load0.max()) + 1e-9
        assert (np.bincount(res.assign, minlength=k) > 0).all()


def test_acgr_improves_bad_partitions():
    g, _ = next(random_graphs(seed=11))
    assign = P.hash_partition(g, 4, seed=0)
    res = ACGR(num_partitions=4).refine_csr(g, assign)
    assert res.cut_after < 0.8 * res.cut_before


def test_legacy_networkx_api():
    G = nx.convert_node_labels_to_integers(nx.planted_partition_graph(4, 30, 0.3, 0.02, seed=0))
    g, _ = CSRGraph.from_networkx(G)
    parts = assign_to_partitions(P.hash_partition(g, 4, seed=0), 4)
    acgr = ACGR(num_partitions=4)
    refined, cost = acgr.refine(G, parts)
    assert len(refined) == 4 and sum(len(p) for p in refined) == G.number_of_nodes()
    assert cost == pytest.approx(acgr.cost(G, refined))
    assert acgr.boundary_cost(G, parts) == 2 * acgr.edge_cut(G, parts)
