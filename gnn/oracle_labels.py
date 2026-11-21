"""
Training data for the runtime controller (outcome-based oracle labels).

For every training stream:
  1. generate a synthetic base graph (SBM, Holme-Kim power-law cluster, random
     geometric or Watts-Strogatz; 3k-25k vertices) and a SyntheticStream with
     randomised regime mix, drift, hotspot and growth parameters
  2. pick a platform (sm / 10g / 1g) and a number of partitions k (4 / 8)
  3. run an epsilon-greedy oracle behaviour policy (eps = 0.3) through the stream;
     at every window the oracle evaluates all four actions with an H-window
     look-ahead on the calibrated cost model
         J(a) = overhead(a) + sum_{h<H} T_exec(window t+h | a now, NO_ACTION afterwards)
     and the label is argmin_a J(a)
  4. store the observation, the label and J(a) for all actions

The real evaluation graphs (Cora, PubMed, sx-mathoverflow) are never used here.

Usage:  python -m gnn.oracle_labels --streams 120 --workers 5
Output: results/controller_data.pt
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np

PLATFORMS = ("sm", "10g", "1g")


def training_graph(rng):
    from graph.csr import CSRGraph

    kind = str(rng.choice(["sbm", "powerlaw", "geometric", "ws"]))
    n = int(rng.integers(3000, 25000))
    seed = int(rng.integers(1 << 30))
    if kind == "sbm":
        c = int(rng.integers(8, 40))
        sizes = np.maximum(20, (rng.dirichlet(np.ones(c) * 2.0) * n).astype(int)).tolist()
        n = sum(sizes)
        d_in, d_out = rng.uniform(4, 12), rng.uniform(0.3, 3.0)
        P = np.full((c, c), d_out / n)
        for i, s in enumerate(sizes):
            P[i, i] = min(1.0, d_in / max(s, 1))
        G = nx.stochastic_block_model(sizes, P.tolist(), seed=seed, sparse=True)
    elif kind == "powerlaw":
        G = nx.powerlaw_cluster_graph(n, int(rng.integers(2, 8)), float(rng.uniform(0.1, 0.6)), seed=seed)
    elif kind == "geometric":
        deg = rng.uniform(5, 14)
        G = nx.random_geometric_graph(n, float(np.sqrt(deg / (np.pi * n))), seed=seed)
    else:
        G = nx.watts_strogatz_graph(n, int(rng.choice([4, 6, 8, 10])), float(rng.uniform(0.01, 0.2)), seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return kind, CSRGraph.from_networkx(G)[0]


def run_stream(job):
    """Worker: one base graph -> one stream -> list of samples (numpy only, picklable)."""
    from controller.policies import EpsilonOracle
    from data.dynamic_stream import SyntheticStream, initial_graph, materialize
    from experiments.common import COST_MODEL
    from experiments.episodes import plan_episode
    from runtime.environment import AdaptiveRuntime, CostModel, RuntimeConfig

    graph_id, seed, windows, horizon, stream_type = job
    rng = np.random.default_rng(seed)
    if stream_type == "temporal":
        from data.dynamic_stream import synthetic_temporal_stream
        stream, _ = synthetic_temporal_stream(rng, num_windows=windows)
        kind = "temporal"
    else:
        kind, base = training_graph(rng)
        stream = make_regime_stream(rng, base, windows)
    platform = str(rng.choice(PLATFORMS))
    k = int(rng.choice([4, 8]))
    cfg = RuntimeConfig(k=k, network=platform, seed=int(rng.integers(1000)))
    rt = AdaptiveRuntime(initial_graph(stream), cfg, cost_model=CostModel.load(COST_MODEL))
    trace = materialize(stream)
    policy = EpsilonOracle(horizon=horizon, eps=0.3, seed=int(rng.integers(1 << 30)))
    records = plan_episode(rt, trace, policy, keep_obs=True)
    return _samples(records, trace, graph_id, kind, platform, k)


def make_regime_stream(rng, base, windows):
    from data.dynamic_stream import SyntheticStream

    return SyntheticStream(
        base, num_windows=windows, seed=int(rng.integers(1 << 30)),
        communities=int(rng.integers(8, 33)),
        p_stay=float(rng.uniform(0.6, 0.9)),
        growth=float(rng.uniform(0.001, 0.006)),
        drift_fraction=float(rng.uniform(0.1, 0.4)),
        hot_fraction=float(rng.uniform(0.3, 0.8)),
        background=float(rng.uniform(0.005, 0.03)),
        regime_weights=tuple(rng.dirichlet(np.ones(3)) + 0.1),
    )


def _samples(records, trace, graph_id, kind, platform, k):
    samples = []
    for r in records:
        obs = r["obs"]
        costs = r["oracle_costs"]
        samples.append(dict(
            graph_id=graph_id, kind=kind, platform=platform, k=k, t=r["t"], regime=r["regime"],
            x=obs.x, globals=obs.globals, edge_index=obs.edge_index, edge_weight=obs.edge_weight,
            work=obs.work, label=r["label"], costs=np.array([costs[a] for a in range(4)]),
            behaviour=r["action"], n=int(trace[r["t"]].g.n),
        ))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=120)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/controller_data.pt")
    ap.add_argument("--stream-type", default="regime", choices=["regime", "temporal"])
    ap.add_argument("--id-offset", type=int, default=0, help="added to graph ids (keeps ids unique across files)")
    args = ap.parse_args()

    import torch

    rng = np.random.default_rng(args.seed)
    jobs = [(args.id_offset + i, int(rng.integers(1 << 30)), args.windows, args.horizon, args.stream_type)
            for i in range(args.streams)]
    samples = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_stream, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            samples.extend(f.result())
            if i % 10 == 0 or i == len(jobs):
                labels = np.bincount([s["label"] for s in samples], minlength=4)
                print(f"{i}/{len(jobs)} streams, {len(samples)} samples, labels={labels.tolist()}, "
                      f"{time.perf_counter() - start:.0f}s", flush=True)
    torch.save({"samples": [to_safe(s) for s in samples], "args": vars(args)}, args.out)
    print("saved", args.out)


# The dataset is stored with tensors and primitive Python types only, so it can be read
# with torch.load(weights_only=True): no pickled objects, no code execution on load.

def to_safe(sample: dict) -> dict:
    import torch

    out = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(np.ascontiguousarray(value))
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def from_safe(sample: dict) -> dict:
    import torch

    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}


def load_samples(path) -> list:
    import torch

    return [from_safe(s) for s in torch.load(path, weights_only=True)["samples"]]


if __name__ == "__main__":
    main()
