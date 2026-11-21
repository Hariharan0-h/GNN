"""
Every number quoted in the paper text, generated from results/ as LaTeX macros
(paper/numbers.tex). The text uses only these macros, so each figure in the paper is
traceable to a result file.
"""

import csv
import json
from collections import defaultdict

import numpy as np

from experiments.common import PAPER, RESULTS


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    macros = {}

    # ---- E4 ----------------------------------------------------------------------
    cal = json.loads((RESULTS / "e4_calibration_summary.json").read_text())
    macros["CalRtwo"] = fmt(cal["r2_test"], 2)
    macros["CalMape"] = fmt(100 * cal["mape_test"], 0)
    macros["CalPoints"] = str(cal["n_points"])
    macros["ReconfMs"] = fmt(1e3 * cal["reconfigure_s"], 1)
    sp = {int(r["threads"]): float(r["speedup"]) for r in csv.DictReader(open(RESULTS / "e4_speedup.csv"))}
    macros["SpeedupSix"] = fmt(sp[6], 1)
    macros["SpeedupMax"] = fmt(max(sp.values()), 1)
    macros["SpeedupMaxThreads"] = str(max(sp, key=sp.get))
    macros["SpeedupTwelve"] = fmt(sp[max(sp)], 1)

    # ---- E2 ----------------------------------------------------------------------
    e2 = json.loads((RESULTS / "e2_controller_metrics.json").read_text())["models"]
    for key, name in [("GS", "GraphSAGE"), ("MLP", "MLP"), ("CE", "GraphSAGE-CE")]:
        macros[f"{key}regret"] = fmt(100 * e2[name]["mean_regret_mean"], 2)
        macros[f"{key}regretStd"] = fmt(100 * e2[name]["mean_regret_std"], 2)
        macros[f"{key}acc"] = fmt(e2[name]["accuracy_mean"], 3)
    for key, name in [("TH", "Threshold"), ("NO", "Always-NO_ACTION"), ("TAall", "Always-TA"),
                      ("RBall", "Always-ACGR"), ("RPall", "Always-METIS")]:
        macros[f"{key}regret"] = fmt(100 * e2[name]["mean_regret"], 2)
    data = json.loads((RESULTS / "e2_controller_metrics.json").read_text())
    for key, name in [("GS", "GraphSAGE"), ("MLP", "MLP")]:
        seeds = e2[name]["per_seed"]
        dist = np.mean([np.array(s["predicted_distribution"]) / sum(s["predicted_distribution"]) for s in seeds], axis=0)
        conf = np.mean([np.array(s["confusion"], dtype=float) for s in seeds], axis=0)
        macros[f"{key}PredRB"] = fmt(100 * dist[2], 0)
        macros[f"{key}PredRP"] = fmt(100 * dist[3], 0)
        macros[f"{key}RBprec"] = fmt(100 * conf[2, 2] / max(conf[:, 2].sum(), 1e-9), 0)
        macros[f"{key}RBrecall"] = fmt(100 * conf[2, 2] / max(conf[2, :].sum(), 1e-9), 0)
        macros[f"{key}RPasRB"] = fmt(100 * conf[3, 2] / max(conf[3, :].sum(), 1e-9), 0)
        macros[f"{key}FOne"] = fmt(e2[name]["macro_f1_mean"], 2)
    macros["THFOne"] = fmt(e2["Threshold"]["macro_f1"], 2)
    counts = data["label_counts"]
    total = sum(sum(v) for v in counts.values())
    lab = np.sum([v for v in counts.values()], axis=0)
    macros["NumSamples"] = f"{total:,}"
    macros["NumTest"] = f"{sum(counts['test']):,}"
    macros["NumTestGraphs"] = str(len(data["split"]["test"]))
    for i, a in enumerate(["NO", "TA", "RB", "RP"]):
        macros[f"Label{a}"] = fmt(100 * lab[i] / total, 1)

    # ---- datasets -----------------------------------------------------------------
    ds = json.loads((RESULTS / "datasets.json").read_text())
    mo = ds["sx-mathoverflow-stream"]
    macros["MathInteractions"] = f"{mo['interactions']:,}".replace(",", "{,}")
    macros["MathYears"] = fmt(mo["span_years"], 1)
    macros["MathActive"] = fmt(100 * mo["mean_active"], 0)
    macros["PubActive"] = fmt(100 * np.mean([ds[f"pubmed-synth-{s}"]["mean_active"] for s in range(3)]), 0)

    # ---- E1 ----------------------------------------------------------------------
    path = RESULTS / "e1_partition_quality.csv"
    if path.exists():
        rows = list(csv.DictReader(open(path)))
        g = defaultdict(list)
        for r in rows:
            g[(r["dataset"], int(r["k"]), r["partitioner"], r["refiner"])].append(r)

        def mean(ds, k, p, ref, col):
            return float(np.mean([float(r[col]) for r in g[(ds, k, p, ref)]]))

        for ds, tag in [("cora", "Cora"), ("pubmed", "Pub"), ("sx-mathoverflow", "Math")]:
            for start in ("LDG", "Hash"):
                cut0, cut1 = mean(ds, 8, start, "none", "edge_cut"), mean(ds, 8, start, "ACGR", "edge_cut")
                lp = mean(ds, 8, start, "LP", "edge_cut")
                macros[f"EOne{tag}{start}CutRed"] = fmt(100 * (1 - cut1 / cut0), 0)
                macros[f"EOne{tag}{start}LPRed"] = fmt(100 * (1 - lp / cut0), 0)
            cv0, cv1 = mean(ds, 8, "METIS", "none", "comm_volume"), mean(ds, 8, "METIS", "ACGR", "comm_volume")
            macros[f"EOne{tag}MetisCVRed"] = fmt(100 * (1 - cv1 / cv0), 1)
            c0, c1 = mean(ds, 8, "METIS", "none", "edge_cut"), mean(ds, 8, "METIS", "ACGR", "edge_cut")
            macros[f"EOne{tag}MetisCutRed"] = fmt(100 * (1 - c1 / c0), 1)
            macros[f"EOne{tag}AcgrMs"] = fmt(1e3 * mean(ds, 8, "METIS", "ACGR", "refine_s"), 0)
            macros[f"EOne{tag}MetisMs"] = fmt(1e3 * mean(ds, 8, "METIS", "none", "partition_s"), 0)

    # ---- E3 ----------------------------------------------------------------------
    path = RESULTS / "e3_summary.csv"
    if path.exists():
        rows = list(csv.DictReader(open(path)))
        g = defaultdict(list)
        for r in rows:
            g[(r["dataset"], int(r["k"]), r["platform"], r["policy"])].append(r)
        DS = {"sx-mathoverflow": "Math", "pubmed-synth": "Pub"}
        KN = {4: "Four", 8: "Eight", 16: "Sixteen"}
        PF = {"sm": "Sm", "10g": "Ten", "1g": "One"}
        PO = {"Static": "Static", "Always-TA": "AlwaysTA", "Always-ACGR": "AlwaysACGR",
              "Periodic-METIS(5)": "PMETIS", "Threshold": "Threshold", "MPC": "MPC", "MLP": "MLP",
              "GraphSAGE": "GSAGE", "Oracle": "Oracle"}
        # robustness: slowdown of each policy relative to the best deployable policy per scenario
        deployable = [p for p in PO if p != "Oracle"]
        scen = sorted({(ds, k, pf) for ds, k, pf, _ in g})
        best = {s: min(np.mean([float(r["total_s"]) for r in g[s + (p,)]]) for p in deployable) for s in scen}
        for p in PO:
            ratios = [np.mean([float(r["total_s"]) for r in g[s + (p,)]]) / best[s] for s in scen]
            macros[f"Rob{PO[p]}Mean"] = fmt(100 * (np.mean(ratios) - 1), 1)
            macros[f"Rob{PO[p]}Max"] = fmt(100 * (np.max(ratios) - 1), 1)
            macros[f"Rob{PO[p]}Wins"] = str(int(sum(np.isclose(r, 1.0) for r in ratios)))
        macros["NumScenarios"] = str(len(scen))
        for p in ("GraphSAGE", "MLP", "Threshold", "Always-TA"):
            sp = [np.mean([float(r["speedup_vs_static"]) for r in g[s + (p,)]]) for s in scen]
            sig = [all(float(r["p_vs_static"]) < 0.05 for r in g[s + (p,)]) for s in scen]
            macros[f"Faster{PO[p]}"] = str(int(sum(x > 1 for x in sp)))
            macros[f"SigFaster{PO[p]}"] = str(int(sum(x > 1 and y for x, y in zip(sp, sig))))
        for p in ("GraphSAGE", "MPC"):
            share = np.mean([float(r["controller_s"]) / float(r["total_s"]) for r in rows if r["policy"] == p])
            macros[f"Ctrl{PO[p]}"] = fmt(100 * share, 1)
        for p in ("GraphSAGE", "Oracle"):
            acts = np.sum([[int(r["n_rb"]), int(r["n_rp"])] for r in rows
                           if r["policy"] == p and r["dataset"] == "sx-mathoverflow"], axis=0)
            macros[f"MathRB{PO[p]}"] = str(int(acts[0]))
            macros[f"MathRP{PO[p]}"] = str(int(acts[1]))
        math_scen = [s for s in scen if s[0] == "sx-mathoverflow"]
        gap = [np.mean([float(r["total_s"]) for r in g[s + ("GraphSAGE",)]]) /
               np.mean([float(r["total_s"]) for r in g[s + ("Oracle",)]]) for s in math_scen]
        macros["RobOracleGap"] = fmt(100 * (max(gap) - 1), 0)
        for (ds, k, pf, pol), rs in g.items():
            key = f"{DS[ds]}{KN[k]}{PF[pf]}{PO[pol]}"
            macros[f"Sp{key}"] = fmt(float(np.mean([float(r["speedup_vs_static"]) for r in rs])), 2)
            macros[f"Tot{key}"] = fmt(float(np.mean([float(r["total_s"]) for r in rs])), 2)

    lines = ["% generated by experiments/numbers.py -- do not edit"]
    for k, v in sorted(macros.items()):
        name = "".join(ch for ch in k if ch.isalpha())
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros to", PAPER / "numbers.tex")


if __name__ == "__main__":
    main()
