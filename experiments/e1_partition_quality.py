"""
E1: static partition quality.

{METIS, KL, LDG, FENNEL, Hash} x {none, LP refinement, ACGR, ACGR-cut ablation}
on Cora, PubMed, DBLP and the final sx-mathoverflow graph, k in {4, 8, 16}, 5 seeds.

ACGR uses eps = 0.03 (the same 3 % tolerance as METIS' default ufactor = 30).
ACGR-cut keeps only the edge-cut term (alpha = 1), isolating the effect of the
communication-volume, balance and skew terms.

Output: results/e1_partition_quality.csv
"""

import argparse
import csv
import time

import numpy as np

from data.graphs import load_static
from experiments.common import RESULTS
from graph import metrics as M
from graph import partitioners as P
from runtime.acgr import ACGR

DATASETS = ("cora", "pubmed", "dblp", "sx-mathoverflow")
PARTITIONERS = ("METIS", "KL", "LDG", "FENNEL", "Hash")
FIELDS = ["dataset", "n", "m", "k", "seed", "partitioner", "refiner", "edge_cut", "cut_ratio",
          "comm_volume", "imbalance_max", "partition_s", "refine_s", "migrated"]


def refiners(k):
    return {
        "none": None,
        "LP": lambda g, a, seed: (P.lp_refine(g, a, k, seed=seed, epsilon=0.03), None),
        "ACGR": lambda g, a, seed: _acgr(g, a, ACGR(num_partitions=k, epsilon=0.03, max_iterations=10)),
        "ACGR-cut": lambda g, a, seed: _acgr(g, a, ACGR(num_partitions=k, alpha=1.0, beta=0.0, gamma=0.0,
                                                       delta=0.0, epsilon=0.03, max_iterations=10)),
    }


def _acgr(g, a, acgr):
    res = acgr.refine_csr(g, a)
    return res.assign, res.time_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()

    # warm up the numba kernels so compilation is not timed
    warm = load_static("cora")
    ACGR(num_partitions=4).refine_csr(warm, P.hash_partition(warm, 4))
    P.lp_refine(warm, P.hash_partition(warm, 4), 4)
    P.ldg(warm, 4)
    P.fennel(warm, 4)

    rows = []
    for name in args.datasets:
        g = load_static(name)
        for k in args.ks:
            for seed in range(args.seeds):
                for part in PARTITIONERS:
                    if part == "KL" and g.m > 100_000:
                        continue  # networkx KL is too slow on the 188k-edge graph
                    t0 = time.perf_counter()
                    base = P.PARTITIONERS[part](g, k, seed=seed)
                    t_part = time.perf_counter() - t0
                    for ref_name, ref in refiners(k).items():
                        if ref is None:
                            assign, t_ref = base, 0.0
                        else:
                            t0 = time.perf_counter()
                            assign, inner = ref(g, base.copy(), seed)
                            t_ref = inner if inner is not None else time.perf_counter() - t0
                        rows.append(dict(
                            dataset=name, n=g.n, m=g.m, k=k, seed=seed, partitioner=part, refiner=ref_name,
                            edge_cut=M.edge_cut(g, assign), cut_ratio=M.edge_cut(g, assign) / g.m,
                            comm_volume=M.comm_volume(g, assign, k),
                            imbalance_max=M.imbalance_max(M.loads(assign, k)),
                            partition_s=t_part, refine_s=t_ref, migrated=int((assign != base).sum()),
                        ))
            print(f"{name} k={k} done ({len(rows)} rows)", flush=True)
    with open(RESULTS / "e1_partition_quality.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print("saved", RESULTS / "e1_partition_quality.csv")


if __name__ == "__main__":
    main()
