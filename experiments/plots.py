"""
Build every paper figure (paper/figs/*.pdf) and table (paper/tables/*.tex) from results/.

Style: vector PDF sized for LNCS (text width 12.2 cm), thin marks, hairline solid grid,
one hue for the learned controllers and neutral gray for baselines (identity is carried
by axis labels, so figures stay readable in grayscale print).
"""

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import PAPER, RESULTS  # noqa: E402

FIGS = PAPER / "figs"
TABLES = PAPER / "tables"
WIDTH = 4.8  # inches, LNCS text width

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4df"
GRAY = "#a3a29c"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference categorical slots 1-4
BLUE_RAMP = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#2a78d6", "#0d3a73"])

POLICY_ORDER = ["Static", "Always-TA", "Always-ACGR", "Periodic-METIS(5)", "Threshold", "MPC", "MLP", "GraphSAGE",
                "Oracle"]
LEARNED = {"MLP", "GraphSAGE"}
PLATFORM_NAME = {"sm": "Shared memory", "10g": "10 GbE (emulated)", "1g": "1 GbE (emulated)"}
DATASET_NAME = {"sx-mathoverflow": "sx-mathoverflow (real)", "pubmed-synth": "PubMed-synthetic"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": GRAY, "axes.linewidth": 0.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
    print("figure", FIGS / f"{name}.pdf")


def write_table(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(text)
    print("table ", TABLES / f"{name}.tex")


# ---------------------------------------------------------------------------
# E4: executor validation
# ---------------------------------------------------------------------------

def fig_e4():
    sp = read_csv(RESULTS / "e4_speedup.csv")
    cal = read_csv(RESULTS / "e4_calibration.csv")
    summary = json.loads((RESULTS / "e4_calibration_summary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 1.65))
    t = np.array([int(r["threads"]) for r in sp])
    s = np.array([float(r["speedup"]) for r in sp])
    a.plot(t, t, color=GRAY, lw=1, label="ideal")
    a.plot(t, s, color=SERIES[0], lw=2, marker="o", ms=3.5, label="measured")
    a.axvline(6, color=GRID, lw=0.8, zorder=0)
    a.text(6.2, 0.6, "6 physical cores", color=INK2, fontsize=6.5)
    a.set_xlabel("threads per partition (k = 1, PubMed)")
    a.set_ylabel("speedup over 1 thread")
    a.set_xlim(0.5, t.max() + 0.5)
    a.set_ylim(0, t.max() + 0.5)
    a.legend(loc="upper left")
    a.set_title("(a) Parallel speedup of the executor", loc="left")
    for i, (ds, name) in enumerate([("pubmed-synth", "PubMed-synthetic"), ("sx-mathoverflow", "sx-mathoverflow")]):
        rows = [r for r in cal if r["dataset"] == ds and r["split"] == "test"]
        m = np.array([float(r["measured_s"]) for r in rows]) * 1e3
        p = np.array([float(r["predicted_s"]) for r in rows]) * 1e3
        b.scatter(m, p, s=9, color=SERIES[i], edgecolor="white", linewidth=0.5, label=name, zorder=3)
    lim = [min(b.get_xlim()[0], b.get_ylim()[0]), max(b.get_xlim()[1], b.get_ylim()[1])]
    b.plot(lim, lim, color=GRAY, lw=1, zorder=1)
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel("measured time, 5 supersteps (ms)")
    b.set_ylabel("predicted (ms)")
    b.set_title(f"(b) Cost model, held-out $R^2$ = {summary['r2_test']:.2f}", loc="left")
    b.legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "e4_executor")


# ---------------------------------------------------------------------------
# E2: controller
# ---------------------------------------------------------------------------

ACTIONS = ["NO", "TA", "RB", "RP"]


def fig_e2():
    res = json.loads((RESULTS / "e2_controller_metrics.json").read_text())
    hist = read_csv(RESULTS / "e2_history.csv")
    cm = np.array(res["models"]["GraphSAGE"]["per_seed"][0]["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 1.5), gridspec_kw={"width_ratios": [1, 1.3]})
    a.imshow(norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
    a.grid(False)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=6.5,
                   color="white" if norm[i, j] > 0.55 else INK)
    a.set_xticks(range(4), ACTIONS)
    a.set_yticks(range(4), ACTIONS)
    a.set_xlabel("predicted")
    a.set_ylabel("oracle label")
    a.set_title("(a) GraphSAGE, test set (seed 0)", loc="left")
    for i, name in enumerate(["GraphSAGE", "MLP"]):
        rows = [r for r in hist if r["model"] == name and r["seed"] == "0"]
        ep = [int(r["epoch"]) for r in rows]
        b.plot(ep, [float(r["train_loss"]) for r in rows], color=SERIES[i], lw=1, alpha=0.45)
        b.plot(ep, [float(r["val_loss"]) for r in rows], color=SERIES[i], lw=2, label=f"{name} (validation)")
    b.set_xlabel("epoch")
    b.set_ylabel("weighted cross-entropy")
    b.set_title("(b) Training (thin) and validation (thick) loss", loc="left")
    b.legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "e2_controller")


def table_e2():
    res = json.loads((RESULTS / "e2_controller_metrics.json").read_text())
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Controller & Accuracy & Macro-F1 & Regret (\%) & Cost ratio \\", r"\midrule"]
    for name in ["GraphSAGE", "MLP"]:
        m = res["models"][name]
        lines.append(f"{name} & {m['accuracy_mean']:.3f} $\\pm$ {m['accuracy_std']:.3f} & "
                     f"{m['macro_f1_mean']:.3f} $\\pm$ {m['macro_f1_std']:.3f} & "
                     f"{100 * m['mean_regret_mean']:.2f} $\\pm$ {100 * m['mean_regret_std']:.2f} & "
                     f"{m['cost_ratio_mean']:.4f} \\\\")
    display = {"Threshold": "Threshold (tuned)", "Always-NO_ACTION": "Never adapt",
               "Always-TA": r"Always \TA{}", "Always-ACGR": r"Always \RB{}", "Always-METIS": r"Always \RP{}"}
    for name in ["Threshold", "Always-NO_ACTION", "Always-TA", "Always-ACGR", "Always-METIS"]:
        m = res["models"][name]
        lines.append(f"{display[name]} & {m['accuracy']:.3f} & {m['macro_f1']:.3f} & {100 * m['mean_regret']:.2f} & "
                     f"{m['cost_ratio']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("e2_controller", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# E3: end-to-end
# ---------------------------------------------------------------------------

def e3_aggregate(rows, k):
    """(dataset, platform, policy) -> (speedup mean, ci half-width) averaged over seeds."""
    groups = defaultdict(list)
    for r in rows:
        if int(r["k"]) != k:
            continue
        groups[(r["dataset"], r["platform"], r["policy"])].append(r)
    out = {}
    for key, rs in groups.items():
        sp = np.array([float(r["speedup_vs_static"]) for r in rs])
        rel = np.array([float(r["total_ci95_s"]) / float(r["total_s"]) for r in rs])
        out[key] = (float(sp.mean()), float(np.mean(sp * rel)))
    return out


def fig_e3(k=4, tag=""):
    path = RESULTS / f"e3_summary{tag}.csv"
    rows = read_csv(path)
    agg = e3_aggregate(rows, k)
    datasets = [d for d in DATASET_NAME if any(key[0] == d for key in agg)]
    platforms = [p for p in PLATFORM_NAME if any(key[1] == p for key in agg)]
    fig, axes = plt.subplots(len(datasets), len(platforms), figsize=(WIDTH, 1.25 + 1.25 * len(datasets)),
                             sharey=True, squeeze=False)
    ypos = np.arange(len(POLICY_ORDER))[::-1]
    for i, ds in enumerate(datasets):
        for j, pf in enumerate(platforms):
            ax = axes[i][j]
            vals = [agg.get((ds, pf, p), (np.nan, 0)) for p in POLICY_ORDER]
            colors = [SERIES[0] if p in LEARNED else ("white" if p == "Oracle" else GRAY) for p in POLICY_ORDER]
            ax.barh(ypos, [v[0] - 1 for v in vals], left=1, height=0.62, color=colors,
                    edgecolor=[SERIES[0] if p in LEARNED else GRAY for p in POLICY_ORDER], linewidth=0.8)
            ax.errorbar([v[0] for v in vals], ypos, xerr=[v[1] for v in vals], fmt="none", ecolor=INK, elinewidth=0.6,
                        capsize=1.5)
            ax.axvline(1.0, color=INK2, lw=0.6)
            ax.grid(axis="y", visible=False)
            ax.set_yticks(ypos, POLICY_ORDER)
            if i == 0:
                ax.set_title(PLATFORM_NAME[pf], loc="center")
            if j == 0:
                ax.set_ylabel(DATASET_NAME[ds])
            if i == len(datasets) - 1:
                ax.set_xlabel("speedup over Static")
            finite = [v[0] for v in vals if np.isfinite(v[0])]
            lo, hi = min(finite + [1.0]), max(finite + [1.0])
            ax.set_xlim(min(0.9, lo - 0.05), hi + 0.08)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, f"e3_speedup_k{k}{tag}")


def table_e3(tag=""):
    rows = read_csv(RESULTS / f"e3_summary{tag}.csv")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], int(r["k"]), r["platform"], r["policy"])].append(r)
    keys = sorted({(d, k, p) for d, k, p, _ in groups}, key=lambda x: (x[0], x[1], list(PLATFORM_NAME).index(x[2])))
    header = " & ".join(["Stream", "$k$", "Platform"] + [p.replace("Periodic-METIS(5)", "P-METIS") for p in POLICY_ORDER])
    lines = [r"\begin{tabular}{lll" + "r" * len(POLICY_ORDER) + "}", r"\toprule", header + r" \\", r"\midrule"]
    for d, k, p in keys:
        cells = []
        totals = {pol: np.mean([float(r["total_s"]) for r in groups[(d, k, p, pol)]])
                  for pol in POLICY_ORDER if (d, k, p, pol) in groups}
        best = min(v for pol, v in totals.items() if pol != "Oracle")
        for pol in POLICY_ORDER:
            rs = groups.get((d, k, p, pol))
            if not rs:
                cells.append("--")
                continue
            mean = totals[pol]
            pval = max(float(r["p_vs_static"]) for r in rs)
            cell = f"{mean:.2f}"
            if pol != "Oracle" and np.isclose(mean, best):
                cell = r"\textbf{" + cell + "}"
            if pol != "Static" and pval < 0.05:
                cell += r"$^{*}$"
            cells.append(cell)
        lines.append(" & ".join([DATASET_NAME[d].split(" ")[0], str(k), p] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(f"e3_totals{tag}", "\n".join(lines) + "\n")


def fig_e3_timeline(dataset="sx-mathoverflow", k=4, platform="1g", tag=""):
    rows = [r for r in read_csv(RESULTS / f"e3_windows{tag}.csv")
            if r["dataset"] == dataset and int(r["k"]) == k and r["platform"] == platform and r["seed"] == "0"]
    pols = ["Static", "Always-TA", "GraphSAGE", "Oracle"]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    markers = {"THREAD_ADJUST": "^", "REBALANCE": "s", "REPARTITION": "D"}
    for i, pol in enumerate(pols):
        rs = sorted([r for r in rows if r["policy"] == pol], key=lambda r: int(r["t"]))
        t = [int(r["t"]) for r in rs]
        y = [float(r["window_s"]) * 1e3 for r in rs]
        ax.plot(t, y, color=SERIES[i], lw=2 if pol == "GraphSAGE" else 1.2, label=pol)
        if pol == "GraphSAGE":
            for act, mk in markers.items():
                pts = [(ti, yi) for ti, yi, r in zip(t, y, rs) if r["action"] == act]
                if pts:
                    ax.scatter(*zip(*pts), marker=mk, s=18, color=SERIES[i], edgecolor="white", linewidth=0.8,
                               zorder=4, label=f"GraphSAGE: {act}")
    ax.set_xlabel("window")
    ax.set_ylabel("window time (ms)")
    ax.legend(ncol=3, loc="upper left", fontsize=6.3)
    ax.set_title(f"{DATASET_NAME[dataset]}, k = {k}, {PLATFORM_NAME[platform]}", loc="left")
    fig.tight_layout()
    save(fig, f"e3_timeline_{dataset}_{platform}{tag}")


# ---------------------------------------------------------------------------
# E1: partition quality
# ---------------------------------------------------------------------------

def table_e1(k=8):
    rows = read_csv(RESULTS / "e1_partition_quality.csv")
    groups = defaultdict(list)
    for r in rows:
        if int(r["k"]) == k:
            groups[(r["dataset"], r["partitioner"], r["refiner"])].append(r)
    datasets = ["cora", "pubmed", "dblp", "sx-mathoverflow"]
    combos = [(p, ref) for p in ["METIS", "KL", "LDG"] for ref in ["none", "LP", "ACGR-cut", "ACGR"]]
    lines = [r"\begin{tabular}{ll" + "rrr" * len(datasets) + "}", r"\toprule",
             "Initial & Refiner & " + " & ".join(r"\multicolumn{3}{c}{" + d + "}" for d in datasets) + r" \\",
             " & & " + " & ".join(["cut \\%", "CV", "ms"] * len(datasets)) + r" \\", r"\midrule"]
    for p, ref in combos:
        cells = []
        for d in datasets:
            rs = groups.get((d, p, ref))
            if not rs:
                cells += ["--"] * 3
                continue
            cells += [f"{100 * np.mean([float(r['cut_ratio']) for r in rs]):.1f}",
                      f"{np.mean([float(r['comm_volume']) for r in rs]):.0f}",
                      f"{1e3 * np.mean([float(r['refine_s']) for r in rs]):.1f}" if ref != "none" else "--"]
        lines.append(f"{p} & {ref} & " + " & ".join(cells) + r" \\")
        if ref == "ACGR":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(f"e1_quality_k{k}", "\n".join(lines) + "\n")


def main():
    style()
    for fn in (fig_e4, fig_e2, table_e2, table_e1, fig_e3, table_e3, fig_e3_timeline):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e.filename} missing")


if __name__ == "__main__":
    main()
