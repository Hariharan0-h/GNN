"""
Partition quality metrics (single source of truth).

edge_cut        number of undirected edges whose endpoints are in different partitions
comm_volume     total number of ghost (halo) copies: sum over vertices v of the number of
                remote partitions that hold at least one neighbour of v. This is the
                communication volume of one BSP halo exchange (Hendrickson & Kolda, 2000).
                Unlike the old "boundary_cost", it is NOT simply 2 x edge_cut.
imbalance_std   std / mean of partition loads (paper Eq. 19)
imbalance_max   max / mean - 1 of partition loads (makespan imbalance)
degree_skew     mean |third standardized moment| of vertex degrees per partition
"""

import numpy as np

from graph.csr import CSRGraph


def edge_cut(g: CSRGraph, assign: np.ndarray) -> int:
    src = g.sources()
    return int((assign[src] != assign[g.indices]).sum() // 2)


def comm_volume(g: CSRGraph, assign: np.ndarray, k: int) -> int:
    src = g.sources()
    p_dst = assign[g.indices]
    mask = p_dst != assign[src]
    if not mask.any():
        return 0
    keys = src[mask] * k + p_dst[mask]
    return int(np.unique(keys).shape[0])


def ghosts_per_partition(g: CSRGraph, assign: np.ndarray, k: int) -> np.ndarray:
    """Number of ghost copies each partition must receive per halo exchange."""
    src = g.sources()
    p_dst = assign[g.indices]
    mask = p_dst != assign[src]
    if not mask.any():
        return np.zeros(k, dtype=np.int64)
    keys = np.unique(src[mask] * k + p_dst[mask])
    return np.bincount(keys % k, minlength=k).astype(np.int64)


def cut_per_partition(g: CSRGraph, assign: np.ndarray, k: int) -> np.ndarray:
    """Number of cut edges incident to each partition."""
    src = g.sources()
    mask = assign[src] != assign[g.indices]
    return np.bincount(assign[src][mask], minlength=k).astype(np.int64)


def loads(assign: np.ndarray, k: int, weights=None) -> np.ndarray:
    return np.bincount(assign, weights=weights, minlength=k).astype(np.float64)


def imbalance_std(load: np.ndarray) -> float:
    mean = float(np.mean(load))
    return 0.0 if mean <= 0 else float(np.std(load) / mean)


def imbalance_max(load: np.ndarray) -> float:
    mean = float(np.mean(load))
    return 0.0 if mean <= 0 else float(np.max(load) / mean - 1.0)


def degree_skew(g: CSRGraph, assign: np.ndarray, k: int) -> float:
    deg = g.degree.astype(np.float64)
    values = []
    for p in range(k):
        d = deg[assign == p]
        if d.size == 0:
            continue
        if d.size <= 1:
            values.append(0.0)
            continue
        std = d.std()
        if std <= 1e-12:
            values.append(0.0)
            continue
        values.append(abs(float(np.mean(((d - d.mean()) / std) ** 3))))
    return float(np.mean(values)) if values else 0.0


def summary(g: CSRGraph, assign: np.ndarray, k: int, weights=None) -> dict:
    count_load = loads(assign, k)
    work_load = loads(assign, k, weights) if weights is not None else count_load
    return {
        "edge_cut": edge_cut(g, assign),
        "comm_volume": comm_volume(g, assign, k),
        "imbalance_std": imbalance_std(count_load),
        "imbalance_max": imbalance_max(count_load),
        "work_imbalance_max": imbalance_max(work_load),
        "degree_skew": degree_skew(g, assign, k),
        "sizes": count_load.astype(int).tolist(),
    }
