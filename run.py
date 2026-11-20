"""
Main entry point for GNN-AdaptGraph

Usage:

python run.py
python run.py --dataset synthetic
python run.py --dataset dblp
"""

import argparse

from gnn.dataset_builder import GraphTrainingDataset
from gnn.datasets.dblp import load_dblp

from controller.runtime_controller import RuntimeController


def run_synthetic():

    print("=" * 60)
    print("Running Synthetic Experiment")
    print("=" * 60)

    dataset = GraphTrainingDataset.build(
        num_graphs=1
    )

    graph = dataset[0]

    controller = RuntimeController()

    actions = controller.predict(graph)

    print("\nRuntime Decisions")
    print("-" * 60)

    for i, action in enumerate(actions):

        print(f"Partition {i+1}: {action}")


def run_dblp():

    print("=" * 60)
    print("Loading DBLP")
    print("=" * 60)

    data = load_dblp()

    print("\nDBLP successfully loaded.")
    print(data)

    print("\nDataset Statistics")
    print("-" * 60)
    print(f"Nodes : {data.num_nodes}")
    print(f"Edges : {data.edge_index.size(1)}")

    print("\nNOTE")
    print("-" * 60)
    print("The Runtime Controller was trained on")
    print("synthetic partition communication graphs.")
    print("The full DBLP runtime pipeline")
    print("(Partition -> Features -> Communication Graph)")
    print("will be integrated next.")


def main():

    parser = argparse.ArgumentParser(
        description="GNN-AdaptGraph"
    )

    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=[
            "synthetic",
            "dblp"
        ],
        help="Dataset to run"
    )

    args = parser.parse_args()

    if args.dataset == "synthetic":

        run_synthetic()

    elif args.dataset == "dblp":

        run_dblp()


if __name__ == "__main__":

    main()