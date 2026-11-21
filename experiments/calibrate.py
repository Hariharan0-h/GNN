"""
Calibrate the execution cost model on this machine (experiment E4).

1. Speedup curve s(tau): one partition, all vertices active, tau = 1..T threads.
2. Random configurations (assignment, threads, window) on PubMed-synthetic and
   sx-mathoverflow windows; fit  T/S = a + b * max_i W_i/s(tau_i) + c * X_comm
   with non-negative least squares; report R^2 on a held-out 30 %.

Outputs: results/cost_model.json, results/e4_speedup.csv, results/e4_calibration.csv
"""

import argparse
import csv
import time

import numpy as np
from scipy.optimize import nnls

from data.dynamic_stream import SyntheticStream, TemporalStream, initial_graph, materialize
from data.graphs import load_static
from experiments.common import COST_MODEL, RESULTS, environment_info, median, write_json
from graph import metrics as M
from graph import partitioners as P
from runtime.bsp_executor import BSPExecutor, allocate_threads, build_layout, even_threads
from runtime.environment import CostModel, RuntimeConfig, active_comm


def speedup_curve(ex, g, total, reps):
    one = np.zeros(g.n, dtype=np.int64)
    lay = build_layout(g, one, 1)
    active = np.ones(g.n, dtype=bool)
    rows = []
    for tau in range(1, total + 1):
        times = [ex.run(g, one, 1, active, [tau], layout=lay)["time_exec"] for _ in range(reps)]
        rows.append((tau, median(times)))
    t1 = rows[0][1]
    return [(tau, t, t1 / t) for tau, t in rows]


def random_configs(trace, g0, k, total, rng, count):
    """Yield (window, assign, tau) with a mix of good and degraded partitions and allocations."""
    base = P.metis(g0, k, seed=0)
    for _ in range(count):
        w = trace[int(rng.integers(len(trace)))]
        kind = rng.integers(4)
        if kind == 0:
            assign = P.metis(w.g, k, seed=int(rng.integers(100)))
        elif kind == 1:
            assign = P.ldg(w.g, k, seed=int(rng.integers(100)))
        elif kind == 2:
            assign = P.hash_partition(w.g, k, seed=int(rng.integers(100)))
        else:
            assign = np.concatenate([base, np.full(w.g.n - base.shape[0], -1)])
            P.ldg_place(w.g, assign, k, np.arange(base.shape[0], w.g.n))
        work = M.loads(assign, k, BSPExecutor.row_work(w.g.degree, 64) * w.active)
        mode = rng.integers(3)
        if mode == 0:
            tau = even_threads(k, total)
        elif mode == 1:
            tau = allocate_threads(work, total)
        else:
            tau = allocate_threads(rng.uniform(0.1, 1.0, k), total)
        yield w, assign, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--configs", type=int, default=120, help="per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = RuntimeConfig()
    steps = 5  # the model is per superstep; short runs keep calibration fast
    ex = BSPExecutor(total_threads=cfg.total_threads, feature_dim=cfg.feature_dim, supersteps=steps)

    # 1. speedup curve
    pubmed = load_static("pubmed")
    curve = speedup_curve(ex, pubmed, cfg.total_threads, args.reps)
    with open(RESULTS / "e4_speedup.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["threads", "median_time_s", "speedup"])
        wr.writerows(curve)
    speedup = [1.0] + [s for _, _, s in curve]
    print("speedup:", [round(s, 2) for s in speedup[1:]])

    # thread-pool reconfiguration time
    reconf = []
    for i in range(10):
        ex.configure_threads([1, 2, 2, 3] if i % 2 else [2, 2, 2, 2])
        start = time.perf_counter()
        ex.configure_threads([2, 2, 2, 2] if i % 2 else [1, 2, 2, 3])
        reconf.append(time.perf_counter() - start)

    # 2. random configurations
    rng = np.random.default_rng(args.seed)
    traces = {
        "pubmed-synth": (lambda s: (initial_graph(s), materialize(s)))(
            SyntheticStream(pubmed, num_windows=12, seed=101)),
        "sx-mathoverflow": (lambda s: (initial_graph(s), materialize(s)))(TemporalStream(num_windows=12)),
    }
    points = []
    for name, (g0, trace) in traces.items():
        for k in (4, 8):
            for w, assign, tau in random_configs(trace, g0, k, cfg.total_threads, rng, args.configs // 2):
                lay = build_layout(w.g, assign, k)
                times = [ex.run(w.g, assign, k, w.active, tau, layout=lay)["time_exec"] for _ in range(3)]
                work = M.loads(assign, k, BSPExecutor.row_work(w.g.degree, cfg.feature_dim) * w.active)
                ghosts, sent, _ = active_comm(w.g, assign, k, w.active)
                points.append(dict(dataset=name, k=k, time=median(times), work=work.tolist(),
                                   tau=list(map(int, tau)), comm=ghosts + sent))
        print(f"{name}: {len(points)} points")
    ex.close()

    # 3. fit
    tmp = CostModel(speedup=speedup)
    X = np.array([[1.0, *tmp.terms(np.array(p["work"]), np.array(p["tau"]), p["comm"])] for p in points])
    y = np.array([p["time"] / steps for p in points])
    idx = rng.permutation(len(points))
    split = int(0.7 * len(points))
    train, test = idx[:split], idx[split:]
    coef, _ = nnls(X[train], y[train])
    model = CostModel(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), speedup=speedup,
                      reconfigure=median(reconf))
    pred = X @ coef

    def r2(ix):
        resid = y[ix] - pred[ix]
        return float(1 - (resid ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum())

    mape = float(np.mean(np.abs(pred[test] - y[test]) / y[test]))
    model.save(COST_MODEL)
    with open(RESULTS / "e4_calibration.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "k", "split", "measured_s", "predicted_s"])
        for i, p in enumerate(points):
            wr.writerow([p["dataset"], p["k"], "train" if i in set(train.tolist()) else "test",
                         p["time"], pred[i] * steps])
    summary = dict(coef=coef.tolist(), r2_train=r2(train), r2_test=r2(test), mape_test=mape,
                   reconfigure_s=median(reconf), n_points=len(points), environment=environment_info())
    write_json(RESULTS / "e4_calibration_summary.json", summary)
    print(f"a={coef[0]:.3e} b={coef[1]:.3e} c={coef[2]:.3e}  R2 train={r2(train):.3f} test={r2(test):.3f} "
          f"MAPE test={mape:.1%}  reconf={median(reconf) * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
