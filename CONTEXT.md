# GNN-AdaptGraph: Project Context & Handoff

Read this first if you are picking up the project. It covers what the system does, how the
code fits together, how to reproduce the paper, and what is known to be limited. The
quick-start is in `README.md`.

## 1. History: why the code was rewritten

The first version (paper 177, 5CSCT 2026) was reviewed in November 2026. Reviewer #2's
concerns were confirmed by reading the code:

| Original claim | What the code actually did |
|---|---|
| Speedups of 1.04× / 1.16× / 0.80× | `runtime/execution_engine.py` ignored the actions. The differences were sub-millisecond noise from a single run. |
| "ACGR overhead caused the PubMed slowdown" | ACGR never ran on PubMed (the controller chose NO_ACTION), and its time was not measured. |
| A boundary-communication term in ACGR | It always equalled 2 × edge cut, so it duplicated the cut term. |
| A reward-driven GraphSAGE policy | The model imitated the hand-written rule in `runtime/decision_engine.py`. |
| THREAD_ADJUST | Not implemented. |
| ACGR improves partitions | It started from KL partitions and penalised all imbalance, so it got stuck after one edge on Cora. |

The revision (branch `revision-5csct`) replaces all of this. The old modules
(`runtime/execution_engine.py`, `runtime/decision_engine.py`, `gnn/model.py`,
`monitor/feature_extractor.py`, the root `test_*.py` scripts, `run_*.py`) are kept only for
reference. None of the new experiments uses them.

## 2. What the system does now

The system is an adaptive runtime for partition-parallel GNN inference on **dynamic graphs**.
In every window:

1. **Ingest** (`runtime/environment.py`): apply the window's edge and vertex changes. New
   vertices are placed with LDG.
2. **Observe**: build 8 partition features, 3 global features and the communication graph.
   One node represents one partition; an edge joins two partitions that share cut edges.
3. **Decide** (`controller/policies.py`, `gnn/controller_model.py`): pick
   NO_ACTION / THREAD_ADJUST / REBALANCE (ACGR) / REPARTITION (METIS scratch-remap).
4. **Apply**: thread reallocation, ACGR, or METIS. Overheads and migration are recorded.
5. **Execute** (`runtime/bsp_executor.py`): run S = 20 supersteps of a GraphSAGE-style layer on
   the active vertices. Each partition has its own thread pool, the kernels are numba kernels
   that release the GIL, and the halo exchange sends only the active ghosts.

**Platforms.** On `sm`, time is fully measured. `10g` and `1g` add an emulated network
charge for the halo exchange and for migration (`NETWORKS` in `runtime/environment.py`).

**Controller training** (`gnn/oracle_labels.py`, `gnn/train_controller.py`):
- Labels come from an oracle that evaluates all 4 actions over an 8-window look-ahead on the
  calibrated cost model (`CostModel`, `results/cost_model.json`).
- The loss is the expected regret plus 0.1 × cross-entropy.
- The data are synthetic training graphs only. The split is by graph instance.

## 3. Key modules

```
graph/csr.py            CSR graph (vertices 0..n-1)
graph/metrics.py        edge cut, communication volume, imbalance, skew (single source of truth)
graph/partitioners.py   METIS (pymetis), KL, LDG, FENNEL, Hash, LP refinement, LDG placement
runtime/acgr.py         ACGR: incremental numba kernel + legacy networkx API
runtime/bsp_executor.py executor, allocate_threads (THREAD_ADJUST policy)
runtime/environment.py  AdaptiveRuntime, CostModel, platforms, oracle look-ahead
data/graphs.py          dataset loaders with an .npz cache (data/cache, git-ignored)
data/dynamic_stream.py  TemporalStream (SNAP sx-mathoverflow), SyntheticStream, materialize()
experiments/            calibrate (E4), e1_partition_quality, e3_dynamic, dataset_stats, plots
tests/                  pytest suite (ACGR incremental == full recompute, executor == scipy)
paper/                  LNCS source; refs.bib is generated from Crossref by make_bib.py
```

## 4. Environment notes

- Use the `.venv` virtual environment. It was created with `--system-site-packages` so that it
  reuses the global torch 2.8.
- **pymetis** has no Windows wheel on PyPI. Install `vendor/pymetis-*.whl`, a patched MSVC
  build; `vendor/README.md` lists the patches.
- **LaTeX**: compile with `powershell -File paper/build.ps1`, which uses TinyTeX in
  `tools/tinytex` (the TinyTeX-v2026.11 zip from GitHub, with `tlmgr install aliascnt
  algorithms algorithmicx multirow booktabs`). Tectonic could not reach its package server from
  this network. `tools/` is git-ignored, so download TinyTeX again if it is missing. Overleaf
  also works: upload the `paper/` folder.
- Do not run timing experiments (E1, E3, E4) while other heavy jobs are running.

## 5. Status of the revision

The revision was completed on 2026-11-21.
- **Paper:** `paper/main.pdf`, 15 pages, LNCS.
- **Response letter:** `paper/response_to_reviewers.md`.
- **Number macros:** every result number in the text is a macro in `paper/numbers.tex`, which is
  generated from `results/`.

Main results:
- **E1 (partition quality):** from streaming partitions, ACGR beats label-propagation
  refinement on the sparse graphs (about 40% cut reduction on PubMed). METIS from scratch stays
  better, and on the dense sx-mathoverflow graph local refinement gains little.
- **E2 (decision quality):** the learned controllers have about 25% lower regret than the tuned
  threshold rule. They select REBALANCE but never REPARTITION.
- **E3 (end-to-end time):**
  - The learned controller is on par with Always-TA and the threshold rule (4.3% from the best
    deployable policy on average), and is not the best in any scenario.
  - Periodic METIS wins on the real stream, and Always-TA wins on PubMed.
  - MPC loses everywhere because of its speculative overhead.
- **Iteration:** the first iteration, trained on regime streams only, never repartitioned at
  all; it is archived in `results/iteration1/`. The second iteration added the synthetic
  interaction streams.

**Open problem:** learning *when to repartition globally*. On the real stream the oracle uses
REBALANCE and REPARTITION and is up to 14% faster than GraphSAGE.

**Measurement note:** the laptop has 8 GB of RAM. Close memory-heavy applications before running
timing experiments. E3 uses medians, because paging once caused 100-second stalls in single runs.

## 6. Known limitations

- Execution is measured on one 6-core laptop. The 10g and 1g platforms emulate network cost
  analytically; there is no real cluster.
- Labels depend on the calibrated cost model. A new machine needs a new calibration and
  retraining.
- The oracle uses a fixed NO_ACTION continuation over H windows. This is a one-step
  look-ahead, not a full planner.
- ACGR is a greedy single-vertex refiner. On dense power-law graphs (sx-mathoverflow) it gains
  much less than METIS.
