import os
import csv
import time
import statistics
import copy

from gnn.datasets.cora import load_cora
from gnn.datasets.dblp import load_dblp
from gnn.datasets.pubmed import load_pubmed

from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.decision_engine import RuntimeDecisionEngine
from controller.runtime_controller import RuntimeController
from runtime.execution_engine import ExecutionEngine

from runtime.acgr import ACGR


CHECKPOINT = ".\\gnn\\checkpoints\\best_graphsage_model.pth"
OUTPUT = ".\\results\\execution_benchmark.csv"

NUM_PARTITIONS = 4
RUNS = 1


# ============================================================
# PARTITION HELPERS
# ============================================================

def prepare_partitions(graph):

    manager = PartitionManager(
        num_partitions=NUM_PARTITIONS
    )

    partitions = manager.partition(graph)

    return normalize_partitions(partitions)


def normalize_partitions(partitions):
    """
    Convert partitions into the required format:

        [
            set(node_ids),
            set(node_ids),
            ...
        ]

    Prevents accidental nested partition structures.
    """

    if partitions is None:
        raise ValueError("Partitions cannot be None")

    # If wrapped once: [[set(...), set(...)]]
    if (
        isinstance(partitions, (list, tuple))
        and len(partitions) == 1
        and isinstance(partitions[0], (list, tuple))
    ):
        partitions = partitions[0]

    normalized = []

    for partition in partitions:

        # Normal partition
        if isinstance(partition, set):
            normalized.append(set(partition))

        elif isinstance(partition, (list, tuple)):

            # Flatten accidental nested set/list
            values = []

            for item in partition:

                if isinstance(item, set):
                    values.extend(list(item))

                elif isinstance(item, (list, tuple)):
                    values.extend(list(item))

                else:
                    values.append(item)

            normalized.append(set(values))

        else:
            raise TypeError(
                "Unsupported partition type: "
                f"{type(partition)}"
            )

    if len(normalized) != NUM_PARTITIONS:
        raise ValueError(
            f"Expected {NUM_PARTITIONS} partitions, "
            f"got {len(normalized)}"
        )

    return normalized


def copy_partitions(partitions):

    return [
        set(partition)
        for partition in partitions
    ]


# ============================================================
# RULE-BASED CONTROLLER
# ============================================================

def rule_based_actions(graph, partitions):

    features = (
        FeatureExtractor
        .extract_partition_features(
            graph,
            partitions
        )
    )

    action_ids = [
        RuntimeDecisionEngine.predict(
            feature
        )
        for feature in features
    ]

    names = {
        0: "NO_ACTION",
        1: "THREAD_ADJUST",
        2: "REBALANCE"
    }

    return [
        names[action]
        for action in action_ids
    ]


# ============================================================
# GRAPHSAGE CONTROLLER
# ============================================================

def graphsage_actions(
    graph,
    partitions,
    controller
):

    features = (
        FeatureExtractor
        .extract_partition_features(
            graph,
            partitions
        )
    )

    from graph.communication_graph import (
        CommunicationGraph
    )

    runtime_graph = (
        CommunicationGraph.build(
            graph,
            partitions,
            features
        )
    )

    return controller.predict(
        runtime_graph
    )


# ============================================================
# ACGR
# ============================================================

def apply_acgr(
    graph,
    partitions
):

    acgr = ACGR(
        num_partitions=NUM_PARTITIONS,

        # Communication-aware configuration
        alpha=0.55,
        beta=0.30,
        gamma=0.10,
        delta=0.05,

        max_iterations=5
    )

    original = copy_partitions(
        partitions
    )

    refined, cost = acgr.refine(
        graph,
        original
    )

    refined = normalize_partitions(
        refined
    )

    return refined, cost


# ============================================================
# SINGLE METHOD EXECUTION
# ============================================================

def run_method(
    graph,
    original_partitions,
    method,
    controller=None
):

    # IMPORTANT:
    # Every method receives a completely independent
    # partition copy.

    partitions = copy_partitions(
        original_partitions
    )

    partitions = normalize_partitions(
        partitions
    )

    # --------------------------------------------------------
    # Static
    # --------------------------------------------------------

    if method == "Static":

        actions = [
            "NO_ACTION"
            for _ in partitions
        ]

    # --------------------------------------------------------
    # Rule-Based
    # --------------------------------------------------------

    elif method == "Rule-Based":

        actions = rule_based_actions(
            graph,
            partitions
        )

    # --------------------------------------------------------
    # GraphSAGE
    # --------------------------------------------------------

    elif method == "GraphSAGE":

        actions = graphsage_actions(
            graph,
            partitions,
            controller
        )

    # --------------------------------------------------------
    # GraphSAGE + ACGR
    # --------------------------------------------------------

    elif method == "GraphSAGE + ACGR":

        # First GraphSAGE decides whether adaptation
        # is required.

        actions = graphsage_actions(
            graph,
            partitions,
            controller
        )

        # Apply ACGR only when GraphSAGE requests
        # repartitioning.

        if "REBALANCE" in actions:

            partitions, acgr_cost = apply_acgr(
                graph,
                partitions
            )

        else:

            acgr_cost = None

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    # --------------------------------------------------------
    # FINAL PARTITION SAFETY CHECK
    # --------------------------------------------------------

    partitions = normalize_partitions(
        partitions
    )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    engine = ExecutionEngine(
        num_partitions=NUM_PARTITIONS
    )

    result = engine.execute(
        graph,
        partitions,
        actions
    )

    return {
        "time": result["execution_time"],
        "processed": result["processed_edges"],
        "throughput": result["throughput"],
        "actions": result["actions"],
        "partitions": result["partitions"]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        ".\\results",
        exist_ok=True
    )

    print("=" * 70)
    print("GNN-AdaptGraph EXECUTION BENCHMARK")
    print("=" * 70)

    print()
    print("Methods:")
    print("  1. Static")
    print("  2. Rule-Based")
    print("  3. GraphSAGE")
    print("  4. GraphSAGE + ACGR")
    print()

    # --------------------------------------------------------
    # Load GraphSAGE Controller
    # --------------------------------------------------------

    controller = RuntimeController(
        CHECKPOINT
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    datasets = [
        ("Cora", load_cora),
        ("DBLP", load_dblp),
        ("PubMed", load_pubmed)
    ]

    methods = [
        "Static",
        "Rule-Based",
        "GraphSAGE",
        "GraphSAGE + ACGR"
    ]

    results = []

    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name, loader in datasets:

        print("\n" + "=" * 70)
        print(dataset_name)
        print("=" * 70)

        graph = loader()

        print(
            f"Nodes : {graph.number_of_nodes()}"
        )

        print(
            f"Edges : {graph.number_of_edges()}"
        )

        # ----------------------------------------------------
        # Initial partitioning
        # ----------------------------------------------------

        base_partitions = prepare_partitions(
            graph
        )

        print(
            f"Partitions: {len(base_partitions)}"
        )

        print(
            "Partition sizes:",
            [
                len(p)
                for p in base_partitions
            ]
        )

        # ====================================================
        # METHOD LOOP
        # ====================================================

        for method in methods:

            print(
                f"\nRunning {method} "
                f"({RUNS} runs)..."
            )

            times = []
            throughputs = []
            processed_edges = []
            action_records = []

            # ------------------------------------------------
            # Warm-up
            # ------------------------------------------------

            try:

                run_method(
                    graph,
                    base_partitions,
                    method,
                    controller
                )

            except Exception as e:

                print(
                    f"Warm-up failed for "
                    f"{method}: {e}"
                )

                raise

            # ------------------------------------------------
            # Measurement
            # ------------------------------------------------

            for run in range(1, RUNS + 1):

                result = run_method(
                    graph,
                    base_partitions,
                    method,
                    controller
                )

                t = result["time"]

                throughput = (
                    result["throughput"]
                )

                processed = (
                    result["processed"]
                )

                actions = (
                    result["actions"]
                )

                times.append(t)
                throughputs.append(
                    throughput
                )
                processed_edges.append(
                    processed
                )
                action_records.append(
                    actions
                )

                print(
                    f"Run {run}: "
                    f"{t:.6f} sec | "
                    f"Throughput: "
                    f"{throughput:.3f} MTEPS | "
                    f"{actions}"
                )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            mean_time = statistics.mean(
                times
            )

            std_time = (
                statistics.stdev(times)
                if len(times) > 1
                else 0.0
            )

            mean_throughput = (
                statistics.mean(
                    throughputs
                )
            )

            std_throughput = (
                statistics.stdev(
                    throughputs
                )
                if len(throughputs) > 1
                else 0.0
            )

            mean_processed = (
                statistics.mean(
                    processed_edges
                )
            )

            print()
            print(
                f"{dataset_name} | "
                f"{method}"
            )

            print(
                f"Mean execution time : "
                f"{mean_time:.6f} sec"
            )

            print(
                f"Std execution time  : "
                f"{std_time:.6f} sec"
            )

            print(
                f"Mean throughput     : "
                f"{mean_throughput:.3f} MTEPS"
            )

            print(
                f"Std throughput      : "
                f"{std_throughput:.3f} MTEPS"
            )

            print(
                f"Mean processed edges: "
                f"{mean_processed:.0f}"
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append({

                "Dataset":
                    dataset_name,

                "Method":
                    method,

                "Mean_Time":
                    mean_time,

                "Std_Time":
                    std_time,

                "Mean_Throughput_MTEPS":
                    mean_throughput,

                "Std_Throughput_MTEPS":
                    std_throughput,

                "Mean_Processed_Edges":
                    mean_processed,

                "Actions":
                    str(action_records)

            })

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "Method",
                "Mean_Time",
                "Std_Time",
                "Mean_Throughput_MTEPS",
                "Std_Throughput_MTEPS",
                "Mean_Processed_Edges",
                "Actions"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL EXECUTION BENCHMARK")
    print("=" * 70)

    for row in results:

        print(
            f"{row['Dataset']:8s} | "
            f"{row['Method']:18s} | "
            f"Time = "
            f"{row['Mean_Time']:.6f} s | "
            f"Throughput = "
            f"{row['Mean_Throughput_MTEPS']:.3f} MTEPS"
        )

    print()
    print("Results saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()