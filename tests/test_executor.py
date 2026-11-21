import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from graph import partitioners as P
from graph.csr import CSRGraph
from runtime.bsp_executor import BSPExecutor, allocate_threads, even_threads, split_by_work


def reference(g, X, W, active, steps):
    n = g.n
    A = sp.csr_matrix((np.ones(g.indices.shape[0]), g.indices, g.indptr), shape=(n, n))
    deg = g.degree.astype(np.float64)
    Dinv = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0))
    X = X.astype(np.float64)
    W = W.astype(np.float64)
    for _ in range(steps):
        agg = Dinv @ (A @ X)
        agg[deg == 0] = X[deg == 0]
        Y = X.copy()
        Y[active] = np.tanh(agg[active] @ W)
        X = Y
    return X


@pytest.mark.parametrize("k,frac", [(1, 1.0), (4, 1.0), (4, 0.4), (8, 0.7)])
def test_executor_matches_reference(k, frac):
    G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(600, 3, 0.2, seed=1))
    G.add_nodes_from(range(600, 610))  # isolated vertices
    g, _ = CSRGraph.from_networkx(G)
    rng = np.random.default_rng(0)
    assign = P.hash_partition(g, k, seed=2) if k > 1 else np.zeros(g.n, dtype=np.int64)
    active = rng.random(g.n) < frac
    tau = allocate_threads(rng.uniform(1, 5, k), 8)
    ex = BSPExecutor(total_threads=8, feature_dim=16, supersteps=5, seed=3)
    try:
        out = ex.run(g, assign, k, active, tau, return_state=True)
    finally:
        ex.close()
    expected = reference(g, ex.initial_state(g.n), ex.weight, active, 5)
    np.testing.assert_allclose(out["state"], expected, rtol=1e-4, atol=1e-5)
    assert out["active"] == int(active.sum())


def test_thread_allocation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 9))
        total = int(rng.integers(k, 17))
        work = rng.uniform(0, 10, k) * (rng.random(k) < 0.8)
        tau = allocate_threads(work, total)
        assert tau.sum() == total and tau.min() >= 1
        assert even_threads(k, total).sum() == total
    tau = allocate_threads(np.array([10.0, 1.0, 1.0, 1.0]), 8)
    assert tau[0] == tau.max() and tau[0] >= 4


def test_split_by_work_covers_rows():
    lens = np.array([5, 1, 1, 1, 10, 2, 2, 3])
    rows = np.arange(8)
    for parts in range(1, 6):
        b = split_by_work(lens, rows, parts)
        assert b[0] == 0 and b[-1] == 8 and (np.diff(b) >= 0).all() and b.shape[0] == parts + 1
