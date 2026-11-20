"""
GNN-AdaptGraph Runtime Execution Experiment

Pipeline:

Graph
    ↓
Partition
    ↓
Runtime Features
    ↓
Communication Graph
    ↓
GraphSAGE Controller
    ↓
Runtime Action per Partition
    ↓
Partition-level Execution
    ↓
Execution Time / Throughput
"""

from gnn.datasets.dblp import load_dblp
from gnn.datasets.cora import load_cora
from gnn.datasets.pubmed import load_pubmed

from controller.runtime_pipeline import RuntimePipeline
from runtime.partition_manager import PartitionManager
from runtime.executor import RuntimeExecutor


# ======================================================
# Configuration
# ======================================================

CHECKPOINT = (
    "gnn/checkpoints/"
    "best_graphsage_model.pth"
)


# ======================================================
# Dataset Runtime Experiment
# ======================================================

def run_dataset(name, loader):

    print("\n" + "=" * 70)
    print(f"{name} RUNTIME EXECUTION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load graph
    # --------------------------------------------------

    graph = loader()

    print(
        f"\nNodes : {graph.number_of_nodes()}"
    )

    print(
        f"Edges : {graph.number_of_edges()}"
    )

    # --------------------------------------------------
    # 2. Runtime controller
    # --------------------------------------------------

    pipeline = RuntimePipeline(
        CHECKPOINT
    )

    actions = pipeline.process(
        graph
    )

    # --------------------------------------------------
    # 3. Partition graph
    # --------------------------------------------------

    partitions = (
        PartitionManager.partition_graph(
            graph,
            4
        )
    )

    # --------------------------------------------------
    # 4. Runtime executor
    # --------------------------------------------------

    executor = RuntimeExecutor()

    action_ids = {
        "NO_ACTION": RuntimeExecutor.NO_ACTION,
        "THREAD_ADJUST": RuntimeExecutor.THREAD_ADJUST,
        "REBALANCE": RuntimeExecutor.REBALANCE
    }

    print(
        "\nExecuting partition-level runtime actions..."
    )

    results = []

    # --------------------------------------------------
    # 5. Execute every partition
    # --------------------------------------------------

    for i, action in enumerate(actions):

        # ----------------------------------------------
        # Convert action label to action ID
        # ----------------------------------------------

        if isinstance(action, str):

            action_id = action_ids[action]

        else:

            action_id = int(action)

        # ----------------------------------------------
        # Execute partition
        # ----------------------------------------------

        result = executor.execute_partition(
            graph,
            partitions[i],
            action_id
        )

        results.append(
            result
        )

        # ----------------------------------------------
        # Display partition result
        # ----------------------------------------------

        print(
            f"\nPartition {i + 1}"
        )

        print(
            f"Nodes           : "
            f"{len(partitions[i])}"
        )

        print(
            f"Action          : "
            f"{result['action']}"
        )

        print(
            f"Threads         : "
            f"{result['threads']}"
        )

        print(
            f"Processed edges : "
            f"{result['processed_edges']}"
        )

        print(
            f"Execution time  : "
            f"{result['execution_time']:.6f} sec"
        )

        print(
            f"Throughput      : "
            f"{result['throughput']:.2f} edges/sec"
        )

    # --------------------------------------------------
    # 6. Aggregate execution results
    # --------------------------------------------------

    total_time = sum(
        result["execution_time"]
        for result in results
    )

    total_processed_edges = sum(
        result["processed_edges"]
        for result in results
    )

    if total_time > 0:

        overall_throughput = (
            total_processed_edges
            / total_time
        )

    else:

        overall_throughput = 0.0

    # --------------------------------------------------
    # 7. Validate edge accounting
    # --------------------------------------------------

    expected_internal_edges = 0

    for partition in partitions:

        node_set = set(partition)

        for u in partition:

            for v in graph.neighbors(u):

                if v in node_set:

                    expected_internal_edges += 1

    # --------------------------------------------------
    # NetworkX undirected graphs visit each edge twice
    #
    # Example:
    #
    # A -> B
    # B -> A
    #
    # Therefore divide by 2.
    # --------------------------------------------------

    internal_edges = (
        expected_internal_edges // 2
    )

    # --------------------------------------------------
    # Communication / cut edges
    # --------------------------------------------------

    communication_edges = (
        graph.number_of_edges()
        - internal_edges
    )

    # --------------------------------------------------
    # 8. Execution summary
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print(
        f"{name} EXECUTION SUMMARY"
    )
    print("-" * 70)

    print(
        f"Graph edges                  : "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Internal partition edges     : "
        f"{internal_edges}"
    )

    print(
        f"Communication/cut edges      : "
        f"{communication_edges}"
    )

    print(
        f"Processed edges              : "
        f"{total_processed_edges}"
    )

    print(
        f"Total execution time         : "
        f"{total_time:.6f} sec"
    )

    print(
        f"Overall throughput           : "
        f"{overall_throughput:.2f} edges/sec"
    )

    # --------------------------------------------------
    # 9. Edge accounting validation
    # --------------------------------------------------

    reconstructed_edges = (
        internal_edges
        + communication_edges
    )

    print(
        f"Reconstructed graph edges    : "
        f"{reconstructed_edges}"
    )

    if (
        reconstructed_edges
        == graph.number_of_edges()
    ):

        print(
            "Edge accounting              : VALID"
        )

    else:

        print(
            "Edge accounting              : WARNING"
        )

    # --------------------------------------------------
    # 10. Action distribution
    # --------------------------------------------------

    action_counts = {
        "NO_ACTION": 0,
        "THREAD_ADJUST": 0,
        "REBALANCE": 0
    }

    for result in results:

        action_name = result["action"]

        if action_name in action_counts:

            action_counts[action_name] += 1

    print(
        "\nAction Distribution"
    )

    print("-" * 40)

    total_partitions = len(results)

    for action_name, count in action_counts.items():

        percentage = (
            count / total_partitions * 100
            if total_partitions > 0
            else 0.0
        )

        print(
            f"{action_name:<18}: "
            f"{count:2d} "
            f"({percentage:6.2f}%)"
        )


# ======================================================
# Main
# ======================================================

print("\n" + "=" * 70)
print(
    "GNN-ADAPTGRAPH RUNTIME EXECUTION EXPERIMENT"
)
print("=" * 70)


# ------------------------------------------------------
# DBLP
# ------------------------------------------------------

run_dataset(
    "DBLP",
    load_dblp
)


# ------------------------------------------------------
# Cora
# ------------------------------------------------------

run_dataset(
    "Cora",
    load_cora
)


# ------------------------------------------------------
# PubMed
# ------------------------------------------------------

run_dataset(
    "PubMed",
    load_pubmed
)


# ======================================================
# Completed
# ======================================================

print("\n" + "=" * 70)
print(
    "RUNTIME EXECUTION EXPERIMENT COMPLETED"
)
print("=" * 70)