import networkx as nx
import numpy as np

from graph import metrics as M
from graph import partitioners as P
from graph.csr import CSRGraph
from runtime.environment import cut_matrix, monitor


def reference_active_comm(g, assign, k, active):
    src = g.sources()
    p_dst = assign[g.indices]
    mask = (p_dst != assign[src]) & active[src]
    if not mask.any():
        return 0, 0, 0
    keys = np.unique(src[mask] * k + p_dst[mask])
    owners = assign[keys // k]
    return int(keys.shape[0]), int(np.unique(keys // k).shape[0]), int(np.unique(owners * k + keys % k).shape[0])


def test_monitor_matches_reference_metrics():
    rng = np.random.default_rng(0)
    for i in range(20):
        G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(int(rng.integers(50, 400)), 3, 0.3, seed=i))
        G.add_nodes_from(range(G.number_of_nodes(), G.number_of_nodes() + 5))
        g, _ = CSRGraph.from_networkx(G)
        k = int(rng.choice([2, 4, 8]))
        assign = P.hash_partition(g, k, seed=i)
        active = rng.random(g.n) < rng.uniform(0.05, 1.0)
        m = monitor(g, assign, k, active)
        np.testing.assert_array_equal(m["ghosts"], M.ghosts_per_partition(g, assign, k))
        np.testing.assert_array_equal(m["mat"], cut_matrix(g, assign, k))
        np.testing.assert_array_equal(m["cut_inc"], M.cut_per_partition(g, assign, k))
        np.testing.assert_array_equal(m["inc"], np.bincount(assign[g.sources()], minlength=k))
        assert (m["act_ghosts"], m["act_sent"], m["pairs"]) == reference_active_comm(g, assign, k, active)
        assert m["mat"].sum() / 2 == M.edge_cut(g, assign)
