"""
Train and evaluate the runtime controllers (experiment E2).

Data      results/controller_data.pt (gnn/oracle_labels.py), split 70/15/15 by base-graph
          instance, so no test graph (or any window of it) is seen during training
Models    GraphSAGEController and MLPController (no message passing), seeds 0, 1, 2
Training  AdamW (lr 1e-3, weight decay 1e-4), batch 64. Cost-sensitive objective:
              L = E_{a ~ softmax(f(s))}[ J(a) / J(a*) - 1 ]  +  0.1 * CE(f(s), a*)
          i.e. the expected regret of the predicted action distribution (every sample
          stores the oracle cost J of all four actions) plus a small cross-entropy term.
          ReduceLROnPlateau (factor 0.5, patience 10); model selection and early stopping
          on the validation regret of the argmax decision (patience 30, max 300 epochs)
Baselines threshold rule (grid-tuned on train+val), constant policies
Metrics   accuracy, macro-F1, confusion matrix and regret:
          regret = J(chosen) / J(oracle) - 1, averaged over test states

Outputs   gnn/checkpoints/controller_<model>_s<seed>.pt
          results/e2_controller_metrics.json, results/e2_history.csv,
          results/threshold_params.json
"""

import argparse
import copy
import csv
import itertools
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from controller.policies import ThresholdPolicy
from experiments.common import RESULTS, ROOT, environment_info, write_json
from gnn.controller_model import MODELS, NUM_ACTIONS, count_parameters

CKPT = ROOT / "gnn" / "checkpoints"
HYPER = dict(lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=300, patience=30,
             lr_factor=0.5, lr_patience=10, hidden=64, dropout=0.3, ce_weight=0.1)


def cost_sensitive_loss(logits, costs, y, ce_weight):
    regret = costs / costs.min(dim=1, keepdim=True).values - 1.0
    expected = (torch.softmax(logits, dim=1) * regret).sum(dim=1).mean()
    return expected + ce_weight * nn.functional.cross_entropy(logits, y)


def to_data(s) -> Data:
    k = s["x"].shape[0]
    glob = np.broadcast_to(s["globals"].reshape(1, -1), (k, s["globals"].shape[0]))
    x = np.ascontiguousarray(np.concatenate([s["x"], glob], axis=1), dtype=np.float32)
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(s["edge_index"]).long(),
        y=torch.tensor([int(s["label"])]),
        costs=torch.from_numpy(s["costs"].astype(np.float32)).view(1, -1),
    )


def split_by_graph(samples, seed):
    ids = np.array(sorted({s["graph_id"] for s in samples}))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr, n_va = int(0.7 * len(ids)), int(0.15 * len(ids))
    groups = {"train": set(ids[:n_tr].tolist()), "val": set(ids[n_tr:n_tr + n_va].tolist()),
              "test": set(ids[n_tr + n_va:].tolist())}
    return {name: [s for s in samples if s["graph_id"] in g] for name, g in groups.items()}


def regret(costs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    chosen = costs[np.arange(costs.shape[0]), actions]
    return chosen / costs.min(axis=1) - 1.0


def evaluate_actions(samples, actions):
    y = np.array([s["label"] for s in samples])
    costs = np.stack([s["costs"] for s in samples])
    return {
        "accuracy": float((actions == y).mean()),
        "macro_f1": float(f1_score(y, actions, average="macro", labels=list(range(NUM_ACTIONS)), zero_division=0)),
        "mean_regret": float(regret(costs, actions).mean()),
        "cost_ratio": float(costs[np.arange(len(y)), actions].sum() / costs.min(axis=1).sum()),
        "confusion": confusion_matrix(y, actions, labels=list(range(NUM_ACTIONS))).tolist(),
        "predicted_distribution": np.bincount(actions, minlength=NUM_ACTIONS).tolist(),
    }


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for batch in loader:
        out.append(torch.argmax(model(batch.x, batch.edge_index, batch.batch), dim=1))
    return torch.cat(out).numpy()


def train_one(name, seed, splits, class_weights, loss="cost"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[name](hidden=HYPER["hidden"], dropout=HYPER["dropout"])
    data = {k: [to_data(s) for s in v] for k, v in splits.items()}
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(data["train"], batch_size=HYPER["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(data["val"], batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=HYPER["lr"], weight_decay=HYPER["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=HYPER["lr_factor"],
                                                       patience=HYPER["lr_patience"])
    def loss_fn(logits, batch):
        if loss == "ce":  # ablation: class-weighted cross-entropy on the oracle labels
            return nn.functional.cross_entropy(logits, batch.y, weight=class_weights)
        return cost_sensitive_loss(logits, batch.costs, batch.y, HYPER["ce_weight"])

    val_costs = np.stack([s["costs"] for s in splits["val"]])
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, HYPER["max_epochs"] + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        model.eval()
        with torch.no_grad():
            va_loss = sum(loss_fn(model(b.x, b.edge_index, b.batch), b).item() * b.num_graphs
                          for b in val_loader) / max(len(data["val"]), 1)
        va_pred = predict(model, val_loader)
        va_acc = float((va_pred == np.array([s["label"] for s in splits["val"]])).mean())
        va_regret = float(regret(val_costs, va_pred).mean())
        sched.step(va_loss)
        history.append(dict(model=name, seed=seed, epoch=epoch, train_loss=tr_loss / n, val_loss=va_loss,
                            val_accuracy=va_acc, val_regret=va_regret, lr=opt.param_groups[0]["lr"]))
        if va_regret < best - 1e-6:
            best, best_state, wait = va_regret, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= HYPER["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history


def tune_threshold(samples):
    drift = np.array([s["x"][:, 4].max() for s in samples])
    pressure = np.array([s["x"][:, 1].max() for s in samples])
    comm = np.array([s["globals"][1] for s in samples])
    costs = np.stack([s["costs"] for s in samples])
    grid_d = np.quantile(drift, np.linspace(0.3, 0.99, 15)).tolist() + [np.inf]
    grid_p = np.quantile(pressure, np.linspace(0.1, 0.99, 15)).tolist() + [np.inf]
    grid_c = np.quantile(comm, np.linspace(0.0, 0.95, 8)).tolist() + [np.inf]
    best = None
    for d, p, c in itertools.product(grid_d, grid_p, grid_c):
        acts = ThresholdPolicy.rule(drift, pressure, comm, d, p, c)
        r = regret(costs, acts).mean()
        if best is None or r < best[0]:
            best = (r, d, p, c)
    return dict(drift_th=float(best[1]), pressure_th=float(best[2]), comm_th=float(best[3]),
                train_mean_regret=float(best[0]))


def threshold_actions(samples, params):
    return ThresholdPolicy.rule(
        np.array([s["x"][:, 4].max() for s in samples]),
        np.array([s["x"][:, 1].max() for s in samples]),
        np.array([s["globals"][1] for s in samples]),
        params["drift_th"], params["pressure_th"], params["comm_th"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(RESULTS / "controller_data.pt"),
                                                  str(RESULTS / "controller_data_temporal.pt")])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    # our own locally generated file (gnn/oracle_labels.py); it holds numpy arrays, hence weights_only=False
    samples = []
    for path in args.data:
        samples.extend(torch.load(path, weights_only=False)["samples"])
    splits = split_by_graph(samples, args.split_seed)
    counts = {k: np.bincount([s["label"] for s in v], minlength=NUM_ACTIONS).tolist() for k, v in splits.items()}
    print("split sizes:", {k: len(v) for k, v in splits.items()}, "labels:", counts)
    y_train = np.array([s["label"] for s in splits["train"]])
    freq = np.bincount(y_train, minlength=NUM_ACTIONS).astype(np.float64)
    cw = np.where(freq > 0, freq.sum() / (NUM_ACTIONS * np.maximum(freq, 1)), 0.0)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    CKPT.mkdir(parents=True, exist_ok=True)
    results = {"hyperparameters": HYPER, "class_weights": cw.tolist(), "label_counts": counts,
               "split": {k: sorted({s["graph_id"] for s in v}) for k, v in splits.items()},
               "environment": environment_info(), "models": {}}
    test_loader = DataLoader([to_data(s) for s in splits["test"]], batch_size=256)
    histories = []
    for name in MODELS:
        per_seed = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            model, hist = train_one(name, seed, splits, class_weights)
            histories.extend(hist)
            m = evaluate_actions(splits["test"], predict(model, test_loader))
            m.update(seed=seed, epochs=len(hist), train_time_s=time.perf_counter() - t0,
                     parameters=count_parameters(model))
            per_seed.append(m)
            torch.save({"model": name, "state_dict": model.state_dict(), "hyperparameters": HYPER},
                       CKPT / f"controller_{name}_s{seed}.pt")
            print(f"{name} seed={seed}: acc={m['accuracy']:.3f} F1={m['macro_f1']:.3f} "
                  f"regret={m['mean_regret']:.4f} epochs={len(hist)} params={m['parameters']}")
        results["models"][name] = {
            "per_seed": per_seed,
            **{f"{key}_mean": float(np.mean([r[key] for r in per_seed])) for key in
               ("accuracy", "macro_f1", "mean_regret", "cost_ratio")},
            **{f"{key}_std": float(np.std([r[key] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
               for key in ("accuracy", "macro_f1", "mean_regret", "cost_ratio")},
        }

    # ablation: same GraphSAGE model trained with class-weighted cross-entropy
    per_seed = []
    for seed in args.seeds:
        model, _ = train_one("GraphSAGE", seed, splits, class_weights, loss="ce")
        m = evaluate_actions(splits["test"], predict(model, test_loader))
        m.update(seed=seed)
        per_seed.append(m)
        print(f"GraphSAGE (weighted CE) seed={seed}: acc={m['accuracy']:.3f} regret={m['mean_regret']:.4f}")
    results["models"]["GraphSAGE-CE"] = {
        "per_seed": per_seed,
        **{f"{key}_mean": float(np.mean([r[key] for r in per_seed])) for key in
           ("accuracy", "macro_f1", "mean_regret", "cost_ratio")},
        **{f"{key}_std": float(np.std([r[key] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
           for key in ("accuracy", "macro_f1", "mean_regret", "cost_ratio")},
    }

    params = tune_threshold(splits["train"] + splits["val"])
    write_json(RESULTS / "threshold_params.json", params)
    results["models"]["Threshold"] = evaluate_actions(splits["test"], threshold_actions(splits["test"], params))
    results["models"]["Threshold"]["params"] = params
    for a, label in enumerate(["Always-NO_ACTION", "Always-TA", "Always-ACGR", "Always-METIS"]):
        results["models"][label] = evaluate_actions(splits["test"], np.full(len(splits["test"]), a))
    for key in ("Threshold", "Always-NO_ACTION", "Always-TA", "Always-ACGR", "Always-METIS"):
        r = results["models"][key]
        print(f"{key}: acc={r['accuracy']:.3f} F1={r['macro_f1']:.3f} regret={r['mean_regret']:.4f}")

    write_json(RESULTS / "e2_controller_metrics.json", results)
    with open(RESULTS / "e2_history.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(histories[0]))
        wr.writeheader()
        wr.writerows(histories)
    print("saved", RESULTS / "e2_controller_metrics.json")


if __name__ == "__main__":
    main()
