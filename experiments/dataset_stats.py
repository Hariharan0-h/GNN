"""Dataset and stream statistics -> results/datasets.json, paper/tables/datasets.tex"""

from collections import Counter

import numpy as np

from data.dynamic_stream import SyntheticStream, TemporalStream, initial_graph, materialize
from data.graphs import load_static, load_temporal
from experiments.common import PAPER, RESULTS, write_json


def main():
    rows, stats = [], {}
    for name, label, kind in [("cora", "Cora", "citation"), ("pubmed", "PubMed", "citation"),
                              ("dblp", "DBLP", "co-authorship"), ("sx-mathoverflow", "sx-mathoverflow (all edges)", "Q\\&A")]:
        g = load_static(name)
        iso = int((g.degree == 0).sum())
        stats[name] = dict(n=g.n, m=g.m, isolated=iso, avg_degree=2 * g.m / g.n)
        rows.append(f"{label} & static & {kind} & {g.n:,} & {g.m:,} & {2 * g.m / g.n:.1f} & -- \\\\")

    src, dst, t, n = load_temporal("sx-mathoverflow")
    span_days = float(t[-1] - t[0]) / 86400.0
    s = TemporalStream(num_windows=30)
    g0, trace = initial_graph(s), materialize(s)
    act = float(np.mean([w.active.mean() for w in trace]))
    stats["sx-mathoverflow-stream"] = dict(
        interactions=int(t.shape[0]), vertices=n, span_days=span_days, span_years=span_days / 365.25,
        initial_n=g0.n, initial_m=g0.m, final_n=trace[-1].g.n, final_m=trace[-1].g.m, mean_active=act,
        min_m=int(min(w.g.m for w in trace)), max_m=int(max(w.g.m for w in trace)))
    rows.append(f"sx-mathoverflow & real stream & Q\\&A & {g0.n:,}$\\to${trace[-1].g.n:,} & "
                f"{g0.m:,}$\\to${trace[-1].g.m:,} & -- & {act:.2f} \\\\")

    base = load_static("pubmed")
    for seed in range(3):
        st = SyntheticStream(base, num_windows=30, seed=1000 + seed)
        tr = materialize(st)
        stats[f"pubmed-synth-{seed}"] = dict(
            final_n=tr[-1].g.n, final_m=tr[-1].g.m, mean_active=float(np.mean([w.active.mean() for w in tr])),
            regimes=dict(Counter(w.regime for w in tr)))
    fin_n = [stats[f"pubmed-synth-{s}"]["final_n"] for s in range(3)]
    fin_m = [stats[f"pubmed-synth-{s}"]["final_m"] for s in range(3)]
    act = np.mean([stats[f"pubmed-synth-{s}"]["mean_active"] for s in range(3)])
    rows.append(f"PubMed-synthetic & synth.\\ stream & citation & {base.n:,}$\\to${int(np.mean(fin_n)):,} & "
                f"{base.m:,}$\\to${int(np.mean(fin_m)):,} & -- & {act:.2f} \\\\")

    write_json(RESULTS / "datasets.json", stats)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "datasets.tex").write_text(
        "\\scriptsize\n\\begin{tabular}{@{}lllrrrr@{}}\n\\toprule\n"
        "Graph & Use & Type & $|V|$ & $|E|$ & avg.\\ deg. & $|A_t|/|V|$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(stats)


if __name__ == "__main__":
    main()
