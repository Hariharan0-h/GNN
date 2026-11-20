# GNN-AdaptGraph: Project Context & Handoff

This document is for whoever picks up this project next. It covers what the project is, how the code fits together, how to run it, where it currently stands, and what is known to be broken or unfinished.

The research write-up is in the repo:
`GNN_AdaptGraph__Adaptive_Graph_Runtime_Optimization_Using_GraphSAGE_and_Communication_Aware_Repartitioning (1).pdf`
(Figures `figure1.png` to `figure5.png` belong to it.)

---

## 1. What the project does

GNN-AdaptGraph is a **runtime controller for partitioned graph processing**. A large graph is split into partitions. A GraphSAGE model looks at those partitions and decides, for each one, which runtime adaptation to apply:

| Label | Action          | Meaning                                         |
|-------|-----------------|-------------------------------------------------|
| 0     | `NO_ACTION`     | Partition is healthy                            |
| 1     | `THREAD_ADJUST` | Moderate imbalance; change parallelism          |
| 2     | `REBALANCE`     | Strong imbalance; repartition (ACGR)            |

When the controller asks for `REBALANCE`, **ACGR** (Adaptive Cost-Aware Graph Repartitioning, `runtime/acgr.py`) greedily moves boundary nodes between partitions. It accepts a move only if the move lowers a weighted cost made up of edge-cut, boundary communication, load imbalance and degree skew.

### End-to-end pipeline (inference)

```
NetworkX graph (Cora / DBLP / PubMed / synthetic)
   │
   ▼  graph/partition_manager.py      recursive Kernighan-Lin bisection → 4 partitions
   │
   ▼  monitor/feature_extractor.py    8 features per partition
   │
   ▼  graph/communication_graph.py    "communication graph": 1 node per partition,
   │                                  edge = cross-partition edges (weight = count)
   │                                  → torch_geometric Data(x=[4,8], edge_index, edge_attr)
   ▼  controller/runtime_controller.py  GraphSAGE → action per partition
   │
   ▼  runtime/acgr.py                 (only if REBALANCE) refine partitions
   │
   ▼  runtime/execution_engine.py     traverse partitions, measure time/throughput
```

`controller/runtime_pipeline.py` (`RuntimePipeline`) runs the first four steps in one call.

### The 8 partition features (`monitor/feature_extractor.py`)

1. Degree skewness  2. Load-imbalance ratio  3. Avg clustering coeff  4. Edge density
5. Avg degree  6. Partition size ratio  7. Boundary edge count  8. Edge-cut ratio

### How the model is trained

- **Data:** `data/synthetic.py` makes random ER, BA or WS graphs with 100 to 300 nodes. `gnn/dataset_builder.py` builds N of them (default 5000; `test_training.py` uses 500).
- **Per graph:** `gnn/graph_dataset.py` partitions it into 4 parts, extracts features and builds the communication graph.
- **Labels** come from the rule-based teacher `runtime/decision_engine.py`:
  `score = 0.40·|skew| + 0.30·imbalance + 0.20·density + 0.10·clustering`, with `≥0.75 → REBALANCE`, `≥0.45 → THREAD_ADJUST`, otherwise `NO_ACTION`.
  **The GraphSAGE model therefore learns to imitate the rule-based engine.** Keep this in mind when you interpret "GraphSAGE vs Rule-Based" comparisons.
- **Model** (`gnn/model.py`): 2× SAGEConv (8→64→64) with BatchNorm, a residual connection, then an MLP head (64→32→3).
- **Trainer** (`gnn/train.py`): AdamW, class-weighted cross-entropy, ReduceLROnPlateau, early stopping (patience 15), 80/10/10 split. Saves the best model to `gnn/checkpoints/best_graphsage_model.pth`.
- **Last recorded test metrics** (`model_metrics.json`): accuracy 0.875, F1 0.868. The `THREAD_ADJUST` class is weak: only 6 of 21 were classified correctly (see the confusion matrix).

---

## 2. Repository layout

```
controller/   runtime_controller.py (loads checkpoint, predicts actions)
              runtime_pipeline.py   (partition → features → comm graph → predict)
gnn/          model.py, train.py, evaluate.py, dataset_builder.py, graph_dataset.py,
              graph_loader.py, preprocessing.py, inference.py, checkpoint.py, graphsage.py
gnn/datasets/ cora.py, dblp.py, pubmed.py, twitter.py   (all return a NetworkX graph)
gnn/checkpoints/ best_graphsage_model.pth  ← checkpoint used by every experiment script
                 best_model.pth
graph/        partition_manager.py, communication_graph.py, feature_extractor.py, graph_builder.py
monitor/      feature_extractor.py (the 8 features), partition_metrics.py, graph_monitor.py
runtime/      acgr.py, decision_engine.py (rule teacher), execution_engine.py, executor.py,
              partition_manager.py, partition_graph.py, rebalance.py, double_buffer.py
evaluation/   latency.py, throughput.py, imbalance.py
config/       runtime_rules.py
data/         synthetic.py, dataset_generator.py, load_graph.py
              (data/Cora, data/DBLP, data/PubMed are auto-downloaded by PyG; git-ignored)
dashboard/    app.py  (Streamlit dashboard)
experiments/  run_experiment.py, results/
results/      CSV outputs of the experiment scripts
checkpoints/  best_graphsage_model.pt  (0 bytes; empty, not used)
```

There are also ~35 `test_*.py` files in the root. **They are standalone scripts, not pytest tests.** Several run work at import time (for example, `test_training.py` trains a model), so do not run `pytest` across the repo.

---

## 3. Setup and running

The original environment used **Python 3.13** on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly          # only needed for the dashboard (not in requirements.txt)
python verify_install.py
```

**Run every script from the repo root.** Imports and paths are relative to it.

| Goal | Command | Output |
|---|---|---|
| Train controller + metrics | `python test_training.py` | `gnn/checkpoints/best_graphsage_model.pth`, `model_metrics.json`, `training_history.json` |
| Pipeline on Cora/DBLP/PubMed | `python run_all_experiments.py` | `results/runtime_results.csv` |
| Rule-based vs GraphSAGE (5 runs) | `python run_fair_comparison.py` | `results/fair_comparison.csv` |
| Static / Rule / GraphSAGE / GraphSAGE+ACGR | `python run_execution_benchmark.py` | `results/execution_benchmark.csv` |
| Baseline comparison | `python run_baseline_comparison.py` | `results/baseline_comparison.csv` |
| Repeated adaptive-vs-static benchmark | `python test_adaptive_comparison.py` | `gnn_adaptgraph_results.csv` |
| ACGR validation | `python validate_acgr.py` | console |
| Dashboard | `streamlit run dashboard/app.py` | browser |
| Simple entry point | `python run.py [--dataset synthetic\|dblp]` | **currently broken, see §4** |

The first run of any real-dataset script downloads the dataset into `data/<Name>/`.

---

## 4. Known issues and open problems

Listed roughly by priority.

1. **`run.py` crashes in both modes.**
   - `run_synthetic()` calls `RuntimeController()` without the required `checkpoint` argument, which raises a `TypeError`. Fix: `RuntimeController("gnn/checkpoints/best_graphsage_model.pth")`.
   - `run_dblp()` treats the return value of `load_dblp()` as a PyG `Data` object (`data.num_nodes`, `data.edge_index`). It is actually a **NetworkX graph**, so this raises an `AttributeError`. Use `G.number_of_nodes()` / `G.number_of_edges()`, or better, run it through `RuntimePipeline` the way `run_all_experiments.py` does.
2. **`main.py` and `README.md` are empty.**
3. **Training and inference use different code paths.**
   - Training (`gnn/graph_dataset.py`) uses `runtime/partition_manager.py` (static, `seed=42`) and `runtime/partition_graph.py`.
   - Inference (`controller/runtime_pipeline.py` and the run scripts) uses `graph/partition_manager.py` (no seed, so results are non-deterministic) and `graph/communication_graph.py`.

   There are also two feature extractors (`monitor/` and `graph/`). Check that these produce the same output, or merge them into one implementation.
4. **Feature issues.** Features 2 and 6 are identical (`len(partition)/total_nodes`). Feature 7 (boundary edge count) and feature 5 (avg degree) are **not normalised**. The model is trained on graphs with 100 to 300 nodes but tested on Cora (2.7k), DBLP (4k) and PubMed (19.7k), so this is a large distribution shift. It probably explains why the controller predicts `REBALANCE` for almost every partition on the real datasets (see `results/fair_comparison.csv`).
5. **`ExecutionEngine` ignores the actions.** Every method does the same Python traversal of intra-partition edges. Runtime differences between Static, Rule-Based and GraphSAGE are timing noise; only ACGR (which really changes the partitions) affects the work done. `THREAD_ADJUST` is not implemented as real parallelism; `test_adaptive_comparison.py` says this explicitly. A real executor (multiprocessing or a native graph library) is needed before any speedup claims hold.
6. **The throughput column labelled "MTEPS" is raw edges/second**, not millions. Divide by 1e6.
7. **`run_execution_benchmark.py` has `RUNS = 1`**, so every std is 0. Increase it for paper-quality numbers.
8. **Hard-coded Windows paths** (`".\\gnn\\checkpoints\\..."`, `".\\results\\..."`) in the run scripts break on Linux/macOS. Switch to `os.path.join` / `pathlib`.
9. `checkpoints/best_graphsage_model.pt` is an empty file. The real checkpoint is `gnn/checkpoints/best_graphsage_model.pth`.
10. The dashboard depends on `streamlit` and `plotly`, which are missing from `requirements.txt`.

---

## 5. Suggested next steps

1. Fix `run.py` (§4.1) so there is one working entry point. Consider making `main.py` that entry point or deleting it.
2. Merge the duplicate partition, feature and communication-graph modules into one path, used by both training and inference.
3. Normalise the features (or train on graphs of realistic size), then retrain and re-check the action distribution on Cora, DBLP and PubMed.
4. Implement real per-action execution (thread/process pools for `THREAD_ADJUST`), then rerun the benchmarks with ≥5 measured runs.
5. Fill in `README.md` with a short summary and quick-start (this file can serve as the source).

---

## 6. Latest results snapshot (`results/`)

- `fair_comparison.csv`: Rule-Based and GraphSAGE choose `REBALANCE` for all partitions on Cora and DBLP. On PubMed, GraphSAGE gives a mixed split (8 NO_ACTION / 4 THREAD_ADJUST / 8 REBALANCE across 5 runs).
- `execution_benchmark.csv`: GraphSAGE + ACGR is fastest on Cora and DBLP. Treat this with caution because of §4.5 and §4.7.
