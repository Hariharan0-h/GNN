import json
import numpy as np

from gnn.dataset_builder import GraphTrainingDataset
from gnn.train import GraphTrainer
from gnn.evaluate import evaluate_model


print("=" * 70)
print("GNN-AdaptGraph GraphSAGE Training Experiment")
print("=" * 70)


# Build training dataset
dataset = GraphTrainingDataset.build(
    num_graphs=500
)


# Train model
trainer = GraphTrainer()

trainer.train(
    dataset=dataset,
    epochs=25
)


# -----------------------------------------
# FINAL TEST EVALUATION
# -----------------------------------------

print("\n")
print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

metrics = evaluate_model(
    trainer.model,
    trainer.test_loader,
    device=str(trainer.device)
)


# -----------------------------------------
# SAVE METRICS
# -----------------------------------------

results = {
    "accuracy": float(metrics["accuracy"]),
    "precision": float(metrics["precision"]),
    "recall": float(metrics["recall"]),
    "f1_score": float(metrics["f1_score"]),
    "confusion_matrix":
        metrics["confusion_matrix"].tolist()
}


with open(
    "model_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# -----------------------------------------
# SAVE TRAINING HISTORY
# -----------------------------------------

history = trainer.history

with open(
    "training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("\nResults saved:")
print("  model_metrics.json")
print("  training_history.json")

print("\nFinal Test Metrics:")
print(
    json.dumps(
        results,
        indent=4
    )
)