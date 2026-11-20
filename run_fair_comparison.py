import os
import csv
import time
import statistics

from gnn.datasets.cora import load_cora
from gnn.datasets.dblp import load_dblp
from gnn.datasets.pubmed import load_pubmed

from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from graph.communication_graph import CommunicationGraph

from runtime.decision_engine import RuntimeDecisionEngine
from controller.runtime_controller import RuntimeController


CHECKPOINT = ".\\gnn\\checkpoints\\best_graphsage_model.pth"
OUTPUT = ".\\results\\fair_comparison.csv"

NUM_RUNS = 5
NUM_PARTITIONS = 4


def action_name(label):
    return {
        0: "NO_ACTION",
        1: "THREAD_ADJUST",
        2: "REBALANCE"
    }[int(label)]


def count_actions(actions):

    return {
        "NO_ACTION": actions.count("NO_ACTION"),
        "THREAD_ADJUST": actions.count("THREAD_ADJUST"),
        "REBALANCE": actions.count("REBALANCE")
    }


def prepare_runtime_graph(graph):

    partition_manager = PartitionManager(
        num_partitions=NUM_PARTITIONS
    )

    feature_extractor = FeatureExtractor()

    communication_builder = CommunicationGraph()

    partitions = partition_manager.partition(graph)

    features = (
        feature_extractor.extract_partition_features(
            graph,
            partitions
        )
    )

    runtime_graph = communication_builder.build(
        graph,
        partitions,
        features
    )

    return features, runtime_graph


def rule_based_decision(features):

    actions = []

    for feature in features:

        label = RuntimeDecisionEngine.predict(
            feature
        )

        actions.append(
            action_name(label)
        )

    return actions


def graphsage_decision(runtime_graph, controller):

    return controller.predict(
        runtime_graph
    )


def run_method(
    graph,
    method,
    controller=None
):

    start = time.perf_counter()

    features, runtime_graph = prepare_runtime_graph(
        graph
    )

    if method == "Rule-Based":

        actions = rule_based_decision(
            features
        )

    else:

        actions = graphsage_decision(
            runtime_graph,
            controller
        )

    end = time.perf_counter()

    return actions, end - start


def main():

    os.makedirs(
        ".\\results",
        exist_ok=True
    )

    print("=" * 70)
    print("GNN-AdaptGraph FAIR CONTROLLER COMPARISON")
    print("=" * 70)

    controller = RuntimeController(
        CHECKPOINT
    )

    datasets = [
        ("Cora", load_cora),
        ("DBLP", load_dblp),
        ("PubMed", load_pubmed)
    ]

    results = []

    for name, loader in datasets:

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        graph = loader()

        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()

        for method in [
            "Rule-Based",
            "GraphSAGE"
        ]:

            print(
                f"\nRunning {method} "
                f"({NUM_RUNS} runs)..."
            )

            times = []

            total_counts = {
                "NO_ACTION": 0,
                "THREAD_ADJUST": 0,
                "REBALANCE": 0
            }

            for run in range(NUM_RUNS):

                actions, elapsed = run_method(
                    graph,
                    method,
                    controller
                )

                times.append(elapsed)

                counts = count_actions(
                    actions
                )

                for key in total_counts:

                    total_counts[key] += counts[key]

                print(
                    f"Run {run + 1}: "
                    f"{elapsed:.6f} sec | "
                    f"{actions}"
                )

            results.append({
                "Dataset": name,
                "Method": method,
                "Nodes": nodes,
                "Edges": edges,
                "Runs": NUM_RUNS,
                "MeanTimeSec":
                    statistics.mean(times),
                "StdTimeSec":
                    statistics.stdev(times),
                "MinTimeSec":
                    min(times),
                "MaxTimeSec":
                    max(times),
                "NoAction":
                    total_counts["NO_ACTION"],
                "ThreadAdjust":
                    total_counts["THREAD_ADJUST"],
                "Rebalance":
                    total_counts["REBALANCE"]
            })

    fields = [
        "Dataset",
        "Method",
        "Nodes",
        "Edges",
        "Runs",
        "MeanTimeSec",
        "StdTimeSec",
        "MinTimeSec",
        "MaxTimeSec",
        "NoAction",
        "ThreadAdjust",
        "Rebalance"
    ]

    with open(
        OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("FAIR COMPARISON COMPLETED")
    print("=" * 70)

    for r in results:

        print(
            f"{r['Dataset']:8s} | "
            f"{r['Method']:12s} | "
            f"{r['MeanTimeSec']:.6f} ± "
            f"{r['StdTimeSec']:.6f} sec"
        )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()