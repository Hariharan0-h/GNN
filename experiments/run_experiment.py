"""
Automated GNN Experiment Runner

Pipeline:
Dataset
   ->
Training
   ->
Best Checkpoint
   ->
Evaluation
   ->
CSV Results
"""


import os
import csv
import torch
import sys


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from gnn.dataset_builder import GraphTrainingDataset
from gnn.dataset_builder import GraphTrainingDataset
from gnn.train import GraphTrainer
from gnn.evaluate import evaluate_model




RESULT_PATH = (
    "experiments/results/results.csv"
)


CHECKPOINT_PATH = (
    "gnn/checkpoints/"
    "best_graphsage_model.pth"
)



def save_results(metrics):

    os.makedirs(
        "experiments/results",
        exist_ok=True
    )


    file_exists = os.path.isfile(
        RESULT_PATH
    )


    with open(
        RESULT_PATH,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)


        if not file_exists:

            writer.writerow(
                [
                    "Dataset",
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1"
                ]
            )


        writer.writerow(
            [
                "Synthetic",
                "GraphSAGE",
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"]
            ]
        )



def main():


    print(
        "\nGenerating Dataset..."
    )


    dataset = GraphTrainingDataset.build(
        num_graphs=500
    )


    print(
        "\nTraining Model..."
    )


    trainer = GraphTrainer()


    trainer.train(
        dataset=dataset,
        epochs=25
    )


    print(
        "\nLoading Best Checkpoint..."
    )


    trainer.model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=trainer.device
        )
    )


    trainer.model.eval()


    print(
        "\nEvaluating..."
    )


    metrics = evaluate_model(
        trainer.model,
        trainer.test_loader
    )


    save_results(
        metrics
    )


    print(
        "\nExperiment Completed"
    )


    print(
        metrics
    )



if __name__ == "__main__":

    main()