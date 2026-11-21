"""
Dataset loaders returning CSRGraph (static) or a timestamped edge stream (temporal).

Static:   cora, pubmed, dblp            (PyTorch Geometric, converted once and cached)
Temporal: sx-mathoverflow               (SNAP, https://snap.stanford.edu/data/sx-mathoverflow.html)

Everything is cached under data/cache/ as .npz so later runs do not need
torch_geometric or the network.
"""

import gzip
import urllib.request
from pathlib import Path

import numpy as np

from graph.csr import CSRGraph

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RAW = ROOT / "raw"

SNAP_URLS = {
    "sx-mathoverflow": "https://snap.stanford.edu/data/sx-mathoverflow.txt.gz",
}

STATIC = ("cora", "pubmed", "dblp")
TEMPORAL = tuple(SNAP_URLS)


def _load_static_networkx(name):
    if name == "cora":
        from gnn.datasets.cora import load_cora
        return load_cora()
    if name == "pubmed":
        from gnn.datasets.pubmed import load_pubmed
        return load_pubmed()
    if name == "dblp":
        from gnn.datasets.dblp import load_dblp
        return load_dblp()
    raise KeyError(name)


def load_static(name: str) -> CSRGraph:
    name = name.lower()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return CSRGraph(indptr=z["indptr"], indices=z["indices"])
    if name in TEMPORAL:
        src, dst, _, n = load_temporal(name)
        g = CSRGraph.from_edges(n, np.stack([src, dst], axis=1))
    else:
        g, _ = CSRGraph.from_networkx(_load_static_networkx(name))
    np.savez_compressed(path, indptr=g.indptr, indices=g.indices)
    return g


def load_temporal(name: str):
    """
    Returns (src, dst, t, n): timestamped edges sorted by time, with vertex ids
    remapped to 0..n-1 in order of first appearance (so vertices arrive in id order).
    Self-loops are dropped; repeated interactions are kept (they are activity).
    """
    name = name.lower()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_temporal.npz"
    if path.exists():
        z = np.load(path)
        return z["src"], z["dst"], z["t"], int(z["n"])
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / Path(SNAP_URLS[name]).name
    if not raw.exists():
        urllib.request.urlretrieve(SNAP_URLS[name], raw)
    with gzip.open(raw, "rt") as f:
        arr = np.loadtxt(f, dtype=np.int64)
    arr = arr[np.argsort(arr[:, 2], kind="mergesort")]
    arr = arr[arr[:, 0] != arr[:, 1]]
    order = np.concatenate([arr[:, 0:1], arr[:, 1:2]], axis=1).ravel()
    _, first_idx = np.unique(order, return_index=True)
    labels_in_order = order[np.sort(first_idx)]
    remap = {int(x): i for i, x in enumerate(labels_in_order.tolist())}
    src = np.fromiter((remap[int(x)] for x in arr[:, 0]), dtype=np.int64, count=arr.shape[0])
    dst = np.fromiter((remap[int(x)] for x in arr[:, 1]), dtype=np.int64, count=arr.shape[0])
    t = arr[:, 2]
    n = len(remap)
    np.savez_compressed(path, src=src, dst=dst, t=t, n=n)
    return src, dst, t, n


def load(name: str) -> CSRGraph:
    return load_static(name)
