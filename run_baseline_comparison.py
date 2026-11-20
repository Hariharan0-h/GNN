import os
import csv
import time
import statistics

from gnn.datasets.cora import load_cora
from gnn.datasets.dblp import load_dblp
from gnn.datasets.pubmed import load_pubmed

from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.decision_engine import RuntimeDecisionEngine

from controller.runtime_pipeline import RuntimePipeline


CHECKPOINT = ".\\gnn\\checkpoints\\best_graphsage_model.pth"
OUTPUT = ".\\results\\baseline_comparison.csv"

NUM_RUNS = 5
NUM_PARTITIONS = 4


def action_name(label):
    mapping = {
        0: "NO_ACTION",
        1: "THREAD_ADJUST",
        2: "REBALANCE"
    }
    return mapping[int(label)]


def count_actions(actions):
    return {
        "NO_ACTION": actions.count("NO_ACTION"),
        "THREAD_ADJUST": actions.count("THREAD_ADJUST"),
        "REBALANCE": actions.count("REBALANCE")
    }


def run_rule_based(graph):
    """
    Rule-based baseline using the existing
    RuntimeDecisionEngine.
    """

    partition_manager = PartitionManager(
        num_partitions=NUM_PARTITIONS
    )

    feature_extractor = FeatureExtractor()

    partitions = partition_manager.partition(graph)

    features = feature_extractor.extract_partition_features(
        graph,
        partitions
    )

    actions = []

    start = time.perf_counter()

    for feature in features:

        label = RuntimeDecisionEngine.predict(
            feature
        )

        actions.append(
            action_name(label)
        )

    end = time.perf_counter()

    return actions, end - start


def run_graphsage(graph, pipeline):
    """
    GNN-AdaptGraph controller.
    """

    start = time.perf_counter()

    actions = pipeline.process(graph)

    end = time.perf_counter()

    return actions, end - start


def summarize(
    dataset,
    method,
    times,
    action_counts,
    nodes,
    edges
):

    return {
        "Dataset": dataset,
        "Method": method,
        "Nodes": nodes,
        "Edges": edges,
        "Runs": len(times),

        "MeanTimeSec":
            statistics.mean(times),

        "StdTimeSec":
            statistics.stdev(times)
            if len(times) > 1 else 0.0,

        "MinTimeSec":
            min(times),

        "MaxTimeSec":
            max(times),

        "NoAction":
            action_counts["NO_ACTION"],

        "ThreadAdjust":
            action_counts["THREAD_ADJUST"],

        "Rebalance":
            action_counts["REBALANCE"]
    }


def main():

    os.makedirs(
        ".\\results",
        exist_ok=True
    )

    print("=" * 70)
    print("GNN-AdaptGraph Baseline Comparison")
    print("=" * 70)

    datasets = [
        ("Cora", load_cora),
        ("DBLP", load_dblp),
        ("PubMed", load_pubmed)
    ]

    pipeline = RuntimePipeline(
        CHECKPOINT
    )

    all_results = []

    for name, loader in datasets:

        print("\n" + "=" * 70)
        print(f"{name}")
        print("=" * 70)

        graph = loader()

        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()

        # --------------------------------------------------
        # Rule-Based Baseline
        # --------------------------------------------------

        print("\nRunning Rule-Based Baseline...")

        rule_times = []

        rule_counts = {
            "NO_ACTION": 0,
            "THREAD_ADJUST": 0,
            "REBALANCE": 0
        }

        for run in range(NUM_RUNS):

            actions, elapsed = run_rule_based(
                graph
            )

            rule_times.append(elapsed)

            counts = count_actions(actions)

            for key in rule_counts:
                rule_counts[key] += counts[key]

            print(
                f"Run {run + 1}: "
                f"{elapsed:.6f} sec | "
                f"{actions}"
            )

        all_results.append(
            summarize(
                name,
                "Rule-Based",
                rule_times,
                rule_counts,
                nodes,
                edges
            )
        )

        # --------------------------------------------------
        # GraphSAGE
        # --------------------------------------------------

        print("\nRunning GraphSAGE Controller...")

        gnn_times = []

        gnn_counts = {
            "NO_ACTION": 0,
            "THREAD_ADJUST": 0,
            "REBALANCE": 0
        }

        for run in range(NUM_RUNS):

            actions, elapsed = run_graphsage(
                graph,
                pipeline
            )

            gnn_times.append(elapsed)

            counts = count_actions(actions)

            for key in gnn_counts:
                gnn_counts[key] += counts[key]

            print(
                f"Run {run + 1}: "
                f"{elapsed:.6f} sec | "
                f"{actions}"
            )

        all_results.append(
            summarize(
                name,
                "GraphSAGE",
                gnn_times,
                gnn_counts,
                nodes,
                edges
            )
        )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    fieldnames = [
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
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL BASELINE COMPARISON")
    print("=" * 70)

    for result in all_results:

        print(
            f"{result['Dataset']:8s} | "
            f"{result['Method']:12s} | "
            f"Mean = "
            f"{result['MeanTimeSec']:.6f} s | "
            f"Std = "
            f"{result['StdTimeSec']:.6f} s"
        )

    print("\nResults saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()