# GNN-AdaptGraph

GNN-AdaptGraph is an adaptive runtime for partition-parallel GNN inference on **dynamic
graphs**. In every window, a learned controller looks at the communication graph of the
current partitioning and picks one of four actions:

| Action | What it does | Cost |
|---|---|---|
| `NO_ACTION` | keeps the current configuration | – |
| `THREAD_ADJUST` | gives each partition threads in proportion to its current work | about 1 ms |
| `REBALANCE` | **ACGR**: local, communication-aware boundary refinement, then threads are re-provisioned | milliseconds; few vertices migrate |
| `REPARTITION` | METIS from scratch, relabelled to maximise overlap with the old partitioning | slower; many vertices migrate |

The controller is a GraphSAGE network trained on **outcome-based oracle labels**. For each
training state, all four actions are evaluated over an 8-window look-ahead on a calibrated
cost model. The training loss is the expected regret.

## Quick start

```bash
python -m venv .venv --system-site-packages     # Python 3.12
.venv\Scripts\activate                          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Windows only: pymetis has no wheel on PyPI, so use the patched local build
pip install --no-deps vendor/pymetis-2025.2.2-cp312-cp312-win_amd64.whl
python -m pytest                                # unit tests (tests/ only)
python run.py --policy GraphSAGE --stream sx-mathoverflow --windows 10 --measure
```

On first use, the datasets are downloaded and cached in `data/cache/`. Cora, PubMed and DBLP
come through PyTorch Geometric; sx-mathoverflow comes from SNAP.

## Reproducing the paper

Run every command from the repository root.

| Step | Command | Output |
|---|---|---|
| E4: calibrate executor and cost model | `python -m experiments.calibrate` | `results/cost_model.json`, `results/e4_*` |
| Training data: regime streams | `python -m gnn.oracle_labels --streams 300` | `results/controller_data.pt` |
| Training data: interaction streams | `python -m gnn.oracle_labels --streams 150 --stream-type temporal --id-offset 100000 --seed 1 --out results/controller_data_temporal.pt` | `results/controller_data_temporal.pt` |
| E2: train and evaluate the controllers | `python -m gnn.train_controller` | `gnn/checkpoints/controller_*.pt`, `results/e2_*` |
| E1: static partition quality | `python -m experiments.e1_partition_quality` | `results/e1_partition_quality.csv` |
| Dataset statistics | `python -m experiments.dataset_stats` | `results/datasets.json`, `paper/tables/datasets.tex` |
| E3: dynamic end-to-end (measured) | `python -m experiments.e3_dynamic` | `results/e3_*.csv` |
| Figures, tables, number macros | `python -m experiments.plots` then `python -m experiments.numbers` | `paper/figs/`, `paper/tables/`, `paper/numbers.tex` |
| Bibliography (needs network) | `python paper/make_bib.py` | `paper/refs.bib` (from Crossref) |
| Paper | `powershell -File paper/build.ps1` (TinyTeX in `tools/`) | `paper/main.pdf` |

Results of the first controller iteration, which was trained on regime streams only, are kept in
`results/iteration1/` and `gnn/checkpoints/iteration1/`. The paper discloses this iteration.

Do not run timing experiments (E1, E3, E4) concurrently with other heavy jobs.

## Layout

```
graph/        csr.py (CSR graph), metrics.py (cut, communication volume, imbalance),
              partitioners.py (METIS, KL, LDG, FENNEL, Hash, label propagation)
runtime/      acgr.py (ACGR), bsp_executor.py (partition-parallel executor),
              environment.py (adaptive runtime, cost model, oracle look-ahead)
data/         graphs.py (dataset loaders), dynamic_stream.py (temporal + synthetic streams)
controller/   policies.py (all compared policies)
gnn/          controller_model.py (GraphSAGE / MLP controllers), oracle_labels.py, train_controller.py
experiments/  calibrate.py (E4), e1_partition_quality.py, e3_dynamic.py, plots.py
tests/        unit tests (pytest)
paper/        LNCS source, refs.bib (generated from Crossref by make_bib.py), figures, tables
```

The `test_*.py` files in the repository root are older standalone scripts from the first
version. They are **not** tests, so do not run them with pytest; `pytest.ini` restricts
collection to `tests/`. `CONTEXT.md` explains how the original version evolved into this one.
