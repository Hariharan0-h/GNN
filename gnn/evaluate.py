import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, loader, device="cpu"):
    """
    Evaluate trained GNN model.

    Returns:
        accuracy
        precision
        recall
        f1
        confusion matrix
    """

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            output = model(
                data.x,
                data.edge_index,
                
            )

            predictions = torch.argmax(
                output,
                dim=1
            )

            y_true.extend(
                data.y.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )


    # Metrics

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    print("="*60)
    print("Evaluation Results")
    print("="*60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }