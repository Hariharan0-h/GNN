"""
GNN-AdaptGraph entry point: run one adaptive episode on a dynamic graph stream.

Examples
  python run.py                                   # sx-mathoverflow, GraphSAGE controller, k=4, shared memory
  python run.py --stream pubmed-synth --policy Threshold --platform 1g
  python run.py --policy Static --measure         # also measure every window on the real executor

The controller checkpoints are produced by `python -m gnn.train_controller` and the
cost model by `python -m experiments.calibrate`.
"""

import argparse
import json

import numpy as np
import torch

from controller import policies as PL
from data.dynamic_stream import SyntheticStream, TemporalStream, initial_graph, materialize
from data.graphs import load_static
from experiments.common import COST_MODEL, RESULTS, ROOT
from experiments.episodes import plan_episode
from gnn.controller_model import MODELS
from runtime.bsp_executor import BSPExecutor
from runtime.environment import AdaptiveRuntime, CostModel, RuntimeConfig


def make_policy(name):
    if name == "Static":
        return PL.static_policy()
    if name == "Always-TA":
        return PL.always_thread_adjust()
    if name == "Always-ACGR":
        return PL.always_rebalance()
    if name == "Periodic-METIS":
        return PL.PeriodicMETIS(5)
    if name == "Oracle":
        return PL.OraclePolicy(8)
    if name == "Threshold":
        p = json.loads((RESULTS / "threshold_params.json").read_text())
        return PL.ThresholdPolicy(p["drift_th"], p["pressure_th"], p["comm_th"])
    ckpt = torch.load(ROOT / "gnn" / "checkpoints" / f"controller_{name}_s0.pt", weights_only=True)
    model = MODELS[name](hidden=ckpt["hyperparameters"]["hidden"], dropout=ckpt["hyperparameters"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return PL.LearnedPolicy(model, name)


def main():
    ap = argparse.ArgumentParser(description="GNN-AdaptGraph")
    ap.add_argument("--stream", default="sx-mathoverflow", choices=["sx-mathoverflow", "pubmed-synth", "cora-synth"])
    ap.add_argument("--policy", default="GraphSAGE",
                    choices=["Static", "Always-TA", "Always-ACGR", "Periodic-METIS", "Threshold", "MLP",
                             "GraphSAGE", "Oracle"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--platform", default="sm", choices=["sm", "10g", "1g"])
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--measure", action="store_true", help="measure each window on the real executor")
    args = ap.parse_args()

    if args.stream == "sx-mathoverflow":
        stream = TemporalStream(num_windows=args.windows)
    else:
        stream = SyntheticStream(load_static(args.stream.split("-")[0]), num_windows=args.windows, seed=0)
    cfg = RuntimeConfig(k=args.k, network=args.platform)
    executor = BSPExecutor(cfg.total_threads, cfg.feature_dim, cfg.supersteps) if args.measure else None
    rt = AdaptiveRuntime(initial_graph(stream), cfg, cost_model=CostModel.load(COST_MODEL), executor=executor)
    trace = materialize(stream)
    records = plan_episode(rt, trace, make_policy(args.policy))

    print(f"{'t':>3} {'regime':8s} {'action':14s} {'predicted':>10s} {'measured':>10s} {'overhead':>9s} {'migrated':>8s}")
    total = 0.0
    for r in records:
        w = trace[r["t"]]
        measured = rt.executor.run(w.g, r["assign"], args.k, w.active, r["tau"])["time_exec"] if args.measure else np.nan
        over = r["adapt_s"] + r["migration_s"] + r["controller_s"]
        total += (measured if args.measure else r["predicted_exec_s"]) + over + rt.network_time_of(r["assign"], w)
        print(f"{r['t']:3d} {r['regime']:8s} {r['action_name']:14s} {r['predicted_exec_s'] * 1e3:8.1f}ms "
              f"{measured * 1e3:8.1f}ms {over * 1e3:7.1f}ms {r['migrated']:8d}")
    print(f"total window time ({'measured' if args.measure else 'predicted'}): {total:.3f} s")
    if executor:
        executor.close()


if __name__ == "__main__":
    main()
