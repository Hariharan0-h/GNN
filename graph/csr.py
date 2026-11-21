"""
Compressed sparse row (CSR) graph used by the partitioners, ACGR,
the executor and the dynamic streams.

Vertices are always the integers 0..n-1. The adjacency is symmetric
(undirected), has no self-loops and no duplicate edges.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class CSRGraph:
    indptr: np.ndarray   # int64, shape [n + 1]
    indices: np.ndarray  # int64, shape [2m]

    @property
    def n(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def m(self) -> int:
        """Number of undirected edges."""
        return int(self.indices.shape[0] // 2)

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def sources(self) -> np.ndarray:
        """Source vertex of every directed adjacency entry (aligned with indices)."""
        return np.repeat(np.arange(self.n, dtype=np.int64), self.degree)

    def edges(self) -> np.ndarray:
        """Undirected edge list with u < v, shape [m, 2]."""
        src = self.sources()
        mask = src < self.indices
        return np.stack([src[mask], self.indices[mask]], axis=1)

    @classmethod
    def from_edges(cls, n: int, edges) -> "CSRGraph":
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if edges.size:
            if edges.min() < 0 or edges.max() >= n:
                raise ValueError("edge endpoint outside 0..n-1")
        u = edges[:, 0]
        v = edges[:, 1]
        keep = u != v
        u, v = u[keep], v[keep]
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        key = np.unique(src * n + dst)
        src = key // n
        dst = key % n
        counts = np.bincount(src, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(indptr=indptr, indices=dst.astype(np.int64))

    @classmethod
    def from_networkx(cls, G: nx.Graph):
        """
        Returns (csr, nodes) where nodes[i] is the networkx label of CSR
        vertex i. When the labels are already 0..n-1 the order is kept.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        if all(isinstance(x, (int, np.integer)) for x in nodes) and set(nodes) == set(range(n)):
            nodes = list(range(n))
            index = None
        else:
            index = {node: i for i, node in enumerate(nodes)}
        if G.number_of_edges() == 0:
            return cls.from_edges(n, np.zeros((0, 2), dtype=np.int64)), nodes
        if index is None:
            edges = np.array(list(G.edges()), dtype=np.int64)
        else:
            edges = np.array([(index[a], index[b]) for a, b in G.edges()], dtype=np.int64)
        return cls.from_edges(n, edges), nodes

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(map(tuple, self.edges().tolist()))
        return G


def partitions_to_assign(partitions, n: int, nodes=None) -> np.ndarray:
    """list[set] -> int64 array of partition ids (CSR vertex order)."""
    assign = np.full(n, -1, dtype=np.int64)
    if nodes is None:
        for pid, part in enumerate(partitions):
            for v in part:
                assign[int(v)] = pid
    else:
        index = {node: i for i, node in enumerate(nodes)}
        for pid, part in enumerate(partitions):
            for v in part:
                assign[index[v]] = pid
    if (assign < 0).any():
        raise ValueError(f"{int((assign < 0).sum())} vertices are not assigned to any partition")
    return assign


def assign_to_partitions(assign: np.ndarray, k: int, nodes=None) -> list:
    """int array -> list[set] (networkx labels when nodes is given)."""
    parts = [set() for _ in range(k)]
    if nodes is None:
        for v, p in enumerate(assign.tolist()):
            parts[p].add(v)
    else:
        for v, p in enumerate(assign.tolist()):
            parts[p].add(nodes[v])
    return parts
