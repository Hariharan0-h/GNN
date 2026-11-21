"""
E3: end-to-end evaluation on dynamic graphs (measured execution).

Streams     sx-mathoverflow (real temporal, 1 stream) and PubMed-synthetic (several seeds).
            Neither graph is used to train the controllers.
Platforms   sm (shared memory), 10g and 1g (emulated network for halo exchange + migration)
Policies    Static, Always-TA, Always-ACGR, Periodic-METIS(5), Threshold (tuned),
            MLP, GraphSAGE (seed-0 checkpoints), MPC (cost model + persistence forecast)
            and the look-ahead Oracle (upper bound)

Protocol
  1. plan: every policy runs through the trace; decisions and overheads (ACGR / METIS
     wall time, controller inference, migration) are recorded
  2. measure: for every window, every distinct configuration (assignment, threads) is
     executed `reps` times on the real executor, policies interleaved in random order
  3. window time = mean measured execution + network time (emulated platforms)
                   + adaptation overhead
Statistics  episode total: mean and 95 % CI over repetitions; paired Wilcoxon
            signed-rank test on per-window times (vs Static and vs the best non-learned
            baseline)

Outputs: results/e3_windows.csv, results/e3_summary.csv
"""

import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch
from scipy.stats import wilcoxon

from controller import policies as PL
from data.dynamic_stream import SyntheticStream, TemporalStream, initial_graph, materialize
from data.graphs import load_static
from experiments.common import COST_MODEL, RESULTS, ROOT, environment_info, mean_ci, write_json
from experiments.episodes import plan_episode
from gnn.controller_model import MODELS
from runtime.bsp_executor import BSPExecutor, build_layout
from runtime.environment import AdaptiveRuntime, CostModel, RuntimeConfig

LEARNED = ("MLP", "GraphSAGE")
BASELINES = ("Static", "Always-TA", "Always-ACGR", "Periodic-METIS(5)", "Threshold", "MPC")


def load_policies(model_seed):
    params = json.loads((RESULTS / "threshold_params.json").read_text())
    pols = [PL.static_policy(), PL.always_thread_adjust(), PL.always_rebalance(), PL.PeriodicMETIS(5),
            PL.ThresholdPolicy(params["drift_th"], params["pressure_th"], params["comm_th"])]
    for name in LEARNED:
        ckpt = torch.load(ROOT / "gnn" / "checkpoints" / f"controller_{name}_s{model_seed}.pt", weights_only=True)
        model = MODELS[name](hidden=ckpt["hyperparameters"]["hidden"], dropout=ckpt["hyperparameters"]["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        pols.append(PL.LearnedPolicy(model, name))
    pols.append(PL.MPCPolicy(horizon=8))
    pols.append(PL.OraclePolicy(horizon=8))
    return pols


def streams(args):
    if "sx-mathoverflow" in args.datasets:
        yield "sx-mathoverflow", 0, TemporalStream(num_windows=args.windows)
    if "pubmed-synth" in args.datasets:
        base = load_static("pubmed")
        for seed in range(args.seeds):
            yield "pubmed-synth", seed, SyntheticStream(base, num_windows=args.windows, seed=1000 + seed)


def config_key(assign, tau):
    return hashlib.sha1(assign.tobytes() + np.asarray(tau, dtype=np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sx-mathoverflow", "pubmed-synth"])
    ap.add_argument("--platforms", nargs="+", default=["sm", "10g", "1g"])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cm = CostModel.load(COST_MODEL)
    cfg0 = RuntimeConfig()
    ex = BSPExecutor(total_threads=cfg0.total_threads, feature_dim=cfg0.feature_dim, supersteps=cfg0.supersteps)
    window_rows, summary_rows = [], []
    rng = np.random.default_rng(0)

    for dataset, seed, stream in streams(args):
        g0, trace = initial_graph(stream), materialize(stream)
        for k in args.ks:
            # ---------------- 1. plan every (platform, policy) ----------------
            plans = {}
            for platform in args.platforms:
                rt = AdaptiveRuntime(g0, RuntimeConfig(k=k, network=platform), cost_model=cm, executor=ex)
                for pol in load_policies(args.model_seed):
                    t0 = time.perf_counter()
                    recs = plan_episode(rt, trace, pol)
                    for r in recs:
                        r["network_s"] = rt.network_time_of(r["assign"], trace[r["t"]])
                    plans[(platform, pol.name)] = recs
                    print(f"[plan] {dataset} s{seed} k={k} {platform} {pol.name}: "
                          f"{time.perf_counter() - t0:.1f}s", flush=True)

            # ---------------- 2. measure distinct configurations ----------------
            t0 = time.perf_counter()
            for t, w in enumerate(trace):
                configs = {}
                for recs in plans.values():
                    key = config_key(recs[t]["assign"], recs[t]["tau"])
                    configs.setdefault(key, (recs[t]["assign"], recs[t]["tau"]))
                layouts = {key: build_layout(w.g, a, k) for key, (a, _) in configs.items()}
                times = {key: [] for key in configs}
                keys = list(configs)
                ex.run(w.g, configs[keys[0]][0], k, w.active, configs[keys[0]][1], layout=layouts[keys[0]])  # warm-up
                for _ in range(args.reps):
                    for key in rng.permutation(keys).tolist():
                        a, tau = configs[key]
                        times[key].append(ex.run(w.g, a, k, w.active, tau, layout=layouts[key])["time_exec"])
                for recs in plans.values():
                    recs[t]["exec_s"] = times[config_key(recs[t]["assign"], recs[t]["tau"])]
            print(f"[measure] {dataset} s{seed} k={k}: {time.perf_counter() - t0:.0f}s", flush=True)

            # ---------------- 3. aggregate ----------------
            for platform in args.platforms:
                per_policy = {}
                for (pf, name), recs in plans.items():
                    if pf != platform:
                        continue
                    over = np.array([r["adapt_s"] + r["migration_s"] + r["controller_s"] for r in recs])
                    net = np.array([r["network_s"] for r in recs])
                    ex_reps = np.array([r["exec_s"] for r in recs])            # [windows, reps]
                    med = np.median(ex_reps, axis=1)
                    # robust per-window execution time: median over repetitions (OS stalls,
                    # e.g. paging, produce rare multi-second outliers; they are counted below)
                    per_window = med + net + over
                    boot_rng = np.random.default_rng(1)
                    idx = boot_rng.integers(0, ex_reps.shape[1], size=(1000,) + ex_reps.shape)
                    boot = np.median(np.take_along_axis(ex_reps[None].repeat(1000, 0), idx, axis=2), axis=2)
                    totals = boot.sum(axis=1) + net.sum() + over.sum()          # bootstrap distribution
                    outliers = int((ex_reps > 3 * med[:, None]).sum())
                    per_policy[name] = (per_window, totals, outliers)
                    for r, pw, m in zip(recs, per_window, med):
                        window_rows.append(dict(
                            dataset=dataset, seed=seed, k=k, platform=platform, policy=name, t=r["t"],
                            regime=r["regime"], action=r["action_name"], exec_median_s=float(m),
                            exec_iqr_s=float(np.subtract(*np.percentile(r["exec_s"], [75, 25]))),
                            network_s=r["network_s"], adapt_s=r["adapt_s"], migration_s=r["migration_s"],
                            controller_s=r["controller_s"], migrated=r["migrated"],
                            predicted_exec_s=r["predicted_exec_s"], window_s=float(pw)))
                static_w = per_policy["Static"][0]
                best_base = min(BASELINES, key=lambda n: per_policy[n][0].sum())
                for name, (pw, totals, outliers) in per_policy.items():
                    total = float(pw.sum())
                    lo, hi = np.percentile(totals, [2.5, 97.5])
                    recs = plans[(platform, name)]
                    acts = [r["action_name"] for r in recs]

                    def p_value(ref):
                        diff = pw - ref
                        return float(wilcoxon(pw, ref).pvalue) if np.any(diff != 0) else 1.0

                    summary_rows.append(dict(
                        dataset=dataset, seed=seed, k=k, platform=platform, policy=name,
                        total_s=total, total_ci95_lo_s=float(lo), total_ci95_hi_s=float(hi),
                        total_ci95_s=float((hi - lo) / 2), outlier_runs=outliers,
                        exec_s=float(sum(np.median(r["exec_s"]) for r in recs)),
                        network_s=float(sum(r["network_s"] for r in recs)),
                        adapt_s=float(sum(r["adapt_s"] for r in recs)),
                        migration_s=float(sum(r["migration_s"] for r in recs)),
                        controller_s=float(sum(r["controller_s"] for r in recs)),
                        migrated=int(sum(r["migrated"] for r in recs)),
                        speedup_vs_static=float(static_w.sum() / pw.sum()),
                        p_vs_static=p_value(static_w), best_baseline=best_base,
                        p_vs_best_baseline=p_value(per_policy[best_base][0]),
                        n_no=acts.count("NO_ACTION"), n_ta=acts.count("THREAD_ADJUST"),
                        n_rb=acts.count("REBALANCE"), n_rp=acts.count("REPARTITION")))
                line = ", ".join(f"{r['policy']}={r['total_s']:.2f}" for r in summary_rows
                                 if r["dataset"] == dataset and r["seed"] == seed and r["k"] == k
                                 and r["platform"] == platform)
                print(f"[result] {dataset} s{seed} k={k} {platform}: {line}", flush=True)
    ex.close()

    suffix = f"_{args.tag}" if args.tag else ""
    for path, rows in [(RESULTS / f"e3_windows{suffix}.csv", window_rows), (RESULTS / f"e3_summary{suffix}.csv", summary_rows)]:
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    write_json(RESULTS / f"e3_meta{suffix}.json", dict(args=vars(args), environment=environment_info(),
                                                       runtime=RuntimeConfig().__dict__))
    print("saved", RESULTS / f"e3_summary{suffix}.csv")


if __name__ == "__main__":
    main()
