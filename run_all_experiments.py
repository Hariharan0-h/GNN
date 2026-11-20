import os
import csv
import time
import networkx as nx

from gnn.datasets.cora import load_cora
from gnn.datasets.dblp import load_dblp
from gnn.datasets.pubmed import load_pubmed

from controller.runtime_pipeline import RuntimePipeline


CHECKPOINT = ".\\gnn\\checkpoints\\best_graphsage_model.pth"
OUTPUT = ".\\results\\runtime_results.csv"


def count_actions(actions):

    return {
        "NO_ACTION": actions.count("NO_ACTION"),
        "THREAD_ADJUST": actions.count("THREAD_ADJUST"),
        "REBALANCE": actions.count("REBALANCE")
    }


def run_dataset(name, loader, pipeline):

    print("\n" + "=" * 60)
    print(f"{name} EXPERIMENT")
    print("=" * 60)

    graph = loader()

    start = time.perf_counter()

    actions = pipeline.process(graph)

    end = time.perf_counter()

    elapsed = end - start

    counts = count_actions(actions)

    print("\nResults")
    print("-" * 60)
    print(f"Dataset              : {name}")
    print(f"Nodes                : {graph.number_of_nodes()}")
    print(f"Edges                : {graph.number_of_edges()}")
    print(f"Partitions            : {len(actions)}")
    print(f"NO_ACTION             : {counts['NO_ACTION']}")
    print(f"THREAD_ADJUST         : {counts['THREAD_ADJUST']}")
    print(f"REBALANCE             : {counts['REBALANCE']}")
    print(f"Pipeline Time (sec)   : {elapsed:.6f}")

    return {
        "Dataset": name,
        "Nodes": graph.number_of_nodes(),
        "Edges": graph.number_of_edges(),
        "Partitions": len(actions),
        "CommunicationEdges": len(actions) * (len(actions) - 1),
        "NoAction": counts["NO_ACTION"],
        "ThreadAdjust": counts["THREAD_ADJUST"],
        "Rebalance": counts["REBALANCE"],
        "PipelineTimeSec": elapsed
    }


def main():

    os.makedirs(".\\results", exist_ok=True)

    pipeline = RuntimePipeline(CHECKPOINT)

    datasets = [
        ("Cora", load_cora),
        ("DBLP", load_dblp),
        ("PubMed", load_pubmed)
    ]

    results = []

    for name, loader in datasets:

        result = run_dataset(
            name,
            loader,
            pipeline
        )

        results.append(result)

    fieldnames = [
        "Dataset",
        "Nodes",
        "Edges",
        "Partitions",
        "CommunicationEdges",
        "NoAction",
        "ThreadAdjust",
        "Rebalance",
        "PipelineTimeSec"
    ]

    with open(
        OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()