"""
GNN-AdaptGraph Controlled Baseline vs Adaptive Execution
==========================================================

Conference-Quality Repeated Benchmark

Methods:
    1. Static Baseline
       - Same graph
       - Same partitions
       - NO_ACTION for every partition

    2. GNN-AdaptGraph
       - Same graph
       - Same partitions
       - GraphSAGE-predicted runtime actions

Measurements:
    - Baseline execution time
    - Adaptive execution time
    - Controller inference time
    - End-to-end adaptive time
    - Runtime-only speedup
    - End-to-end speedup
    - Throughput
    - Mean +/- standard deviation
    - Action distribution
    - Edge accounting

Experimental protocol:
    - Warm-up runs are excluded from measurements.
    - Multiple measured runs are used.
    - Graph and partitioning are created once.
    - The trained GraphSAGE controller is loaded once.
    - Controller inference is measured separately.
    - Runtime execution is measured separately.

IMPORTANT:
    RuntimeExecutor currently uses Python traversal.
    Therefore THREAD_ADJUST is treated as a runtime
    configuration label rather than true parallel execution.
"""

from gnn.datasets.dblp import load_dblp
from gnn.datasets.cora import load_cora
from gnn.datasets.pubmed import load_pubmed

from controller.runtime_pipeline import RuntimePipeline
from runtime.partition_manager import PartitionManager
from runtime.executor import RuntimeExecutor

import multiprocessing
import time
import statistics
import csv
import os


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "gnn/checkpoints/"
    "best_graphsage_model.pth"
)

NUM_PARTITIONS = 4

# ------------------------------------------------------------
# Repeated benchmark configuration
# ------------------------------------------------------------

WARMUP_RUNS = 3
MEASUREMENT_RUNS = 10

RESULTS_FILE = "gnn_adaptgraph_results.csv"


# ============================================================
# EXECUTE PARTITIONS
# ============================================================

def execute_partitions(
    graph,
    partitions,
    actions,
    executor
):
    """
    Execute every partition using supplied actions.
    """

    results = []

    for i, action in enumerate(actions):

        result = executor.execute_partition(
            graph,
            partitions[i],
            action
        )

        results.append(result)

    total_time = sum(
        r["execution_time"]
        for r in results
    )

    total_edges = sum(
        r["processed_edges"]
        for r in results
    )

    throughput = (
        total_edges / total_time
        if total_time > 0
        else 0.0
    )

    return {
        "results": results,
        "total_time": total_time,
        "total_edges": total_edges,
        "throughput": throughput
    }


# ============================================================
# ACTION CONVERSION
# ============================================================

def convert_actions(actions):
    """
    Convert controller actions to RuntimeExecutor IDs.
    """

    action_ids = {
        "NO_ACTION": RuntimeExecutor.NO_ACTION,
        "THREAD_ADJUST": RuntimeExecutor.THREAD_ADJUST,
        "REBALANCE": RuntimeExecutor.REBALANCE
    }

    converted = []

    for action in actions:

        if isinstance(action, str):

            if action not in action_ids:

                raise ValueError(
                    f"Unknown action: {action}"
                )

            converted.append(
                action_ids[action]
            )

        else:

            converted.append(
                int(action)
            )

    return converted


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

def get_action_distribution(results):

    counts = {
        "NO_ACTION": 0,
        "THREAD_ADJUST": 0,
        "REBALANCE": 0
    }

    for result in results:

        action = result["action"]

        if action in counts:
            counts[action] += 1

    return counts


def print_action_distribution(
    results,
    title
):

    counts = get_action_distribution(results)

    total = len(results)

    print(f"\n{title}")
    print("-" * 40)

    for action, count in counts.items():

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"{action:<18}: "
            f"{count} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# EDGE ACCOUNTING
# ============================================================

def calculate_internal_edges(
    graph,
    partitions
):
    """
    Count adjacency traversals inside partitions.

    RuntimeExecutor counts each adjacency traversal
    exactly once. Therefore no multiplication by 2.
    """

    internal_edges = 0

    for partition in partitions:

        node_set = set(partition)

        for u in partition:

            for v in graph.neighbors(u):

                if v in node_set:

                    internal_edges += 1

    return internal_edges


# ============================================================
# STATISTICS
# ============================================================

def mean(values):

    if not values:
        return 0.0

    return statistics.mean(values)


def std(values):

    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def min_value(values):

    if not values:
        return 0.0

    return min(values)


def max_value(values):

    if not values:
        return 0.0

    return max(values)


# ============================================================
# RUN BASELINE
# ============================================================

def run_baseline_trials(
    graph,
    partitions,
    executor
):

    actions = [
        RuntimeExecutor.NO_ACTION
        for _ in partitions
    ]

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nRunning {WARMUP_RUNS} baseline warm-up runs..."
    )

    for _ in range(WARMUP_RUNS):

        execute_partitions(
            graph,
            partitions,
            actions,
            executor
        )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    times = []
    throughputs = []
    edge_counts = []

    print(
        f"Running {MEASUREMENT_RUNS} baseline "
        f"measurement runs..."
    )

    for run in range(MEASUREMENT_RUNS):

        result = execute_partitions(
            graph,
            partitions,
            actions,
            executor
        )

        times.append(
            result["total_time"]
        )

        throughputs.append(
            result["throughput"]
        )

        edge_counts.append(
            result["total_edges"]
        )

        print(
            f"  Baseline run "
            f"{run + 1:02d}: "
            f"{result['total_time']:.6f} sec"
        )

    return {
        "actions": actions,
        "times": times,
        "throughputs": throughputs,
        "edges": edge_counts
    }


# ============================================================
# RUN CONTROLLER INFERENCE
# ============================================================

def run_controller_trials(
    pipeline,
    graph
):

    # --------------------------------------------------------
    # Warm-up inference
    # --------------------------------------------------------

    print(
        f"\nRunning {WARMUP_RUNS} controller warm-up runs..."
    )

    warmup_actions = None

    for _ in range(WARMUP_RUNS):

        warmup_actions = pipeline.process(
            graph
        )

    # --------------------------------------------------------
    # Measured inference
    # --------------------------------------------------------

    inference_times = []

    predicted_actions = None

    print(
        f"Running {MEASUREMENT_RUNS} "
        f"controller measurement runs..."
    )

    for run in range(MEASUREMENT_RUNS):

        start = time.perf_counter()

        predicted_actions = pipeline.process(
            graph
        )

        end = time.perf_counter()

        inference_time = end - start

        inference_times.append(
            inference_time
        )

        print(
            f"  Controller run "
            f"{run + 1:02d}: "
            f"{inference_time:.6f} sec"
        )

    return {
        "times": inference_times,
        "actions": predicted_actions
    }


# ============================================================
# RUN ADAPTIVE EXECUTION
# ============================================================

def run_adaptive_trials(
    graph,
    partitions,
    adaptive_actions,
    executor
):

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nRunning {WARMUP_RUNS} adaptive warm-up runs..."
    )

    for _ in range(WARMUP_RUNS):

        execute_partitions(
            graph,
            partitions,
            adaptive_actions,
            executor
        )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    times = []
    throughputs = []
    edge_counts = []

    print(
        f"Running {MEASUREMENT_RUNS} adaptive "
        f"measurement runs..."
    )

    for run in range(MEASUREMENT_RUNS):

        result = execute_partitions(
            graph,
            partitions,
            adaptive_actions,
            executor
        )

        times.append(
            result["total_time"]
        )

        throughputs.append(
            result["throughput"]
        )

        edge_counts.append(
            result["total_edges"]
        )

        print(
            f"  Adaptive run "
            f"{run + 1:02d}: "
            f"{result['total_time']:.6f} sec"
        )

    return {
        "times": times,
        "throughputs": throughputs,
        "edges": edge_counts
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    title,
    times,
    throughputs
):

    print("\n" + title)
    print("-" * 55)

    print(
        f"Mean time       : "
        f"{mean(times):.6f} sec"
    )

    print(
        f"Std. deviation  : "
        f"{std(times):.6f} sec"
    )

    print(
        f"Minimum time     : "
        f"{min_value(times):.6f} sec"
    )

    print(
        f"Maximum time     : "
        f"{max_value(times):.6f} sec"
    )

    print(
        f"Mean throughput : "
        f"{mean(throughputs):.2f} edges/sec"
    )

    print(
        f"Throughput std  : "
        f"{std(throughputs):.2f} edges/sec"
    )


# ============================================================
# SAVE CSV RESULT
# ============================================================

def save_result_row(row):

    file_exists = os.path.exists(
        RESULTS_FILE
    )

    fieldnames = [
        "dataset",
        "nodes",
        "graph_edges",
        "partitions",

        "baseline_mean_time",
        "baseline_std_time",
        "baseline_mean_throughput",

        "controller_mean_time",
        "controller_std_time",

        "adaptive_mean_time",
        "adaptive_std_time",
        "adaptive_mean_throughput",

        "end_to_end_mean_time",

        "runtime_speedup",
        "end_to_end_speedup",

        "throughput_gain",
        "throughput_improvement_percent",

        "processed_edges",
        "expected_edges",

        "edge_validation",

        "no_action",
        "thread_adjust",
        "rebalance"
    ]

    with open(
        RESULTS_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(row)


# ============================================================
# MAIN DATASET EXPERIMENT
# ============================================================

def run_dataset(
    name,
    loader
):

    print("\n" + "=" * 80)
    print(
        f"{name} REPEATED CONTROLLED "
        f"BASELINE VS GNN-ADAPTGRAPH"
    )
    print("=" * 80)

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    graph = loader()

    nodes = graph.number_of_nodes()
    graph_edges = graph.number_of_edges()

    print(
        f"\nNodes : {nodes}"
    )

    print(
        f"Edges : {graph_edges}"
    )

    # ========================================================
    # COMMON PARTITIONS
    # ========================================================

    print(
        "\nCreating common partitions..."
    )

    partitions = (
        PartitionManager.partition_graph(
            graph,
            NUM_PARTITIONS
        )
    )

    print(
        f"Partitions : {len(partitions)}"
    )

    for i, partition in enumerate(partitions):

        print(
            f"Partition {i + 1}: "
            f"{len(partition)} nodes"
        )

    # ========================================================
    # EDGE ACCOUNTING
    # ========================================================

    internal_edges = calculate_internal_edges(
        graph,
        partitions
    )

    expected_edges = internal_edges

    print(
        f"\nExpected processed edges : "
        f"{expected_edges}"
    )

    # ========================================================
    # EXECUTOR
    # ========================================================

    executor = RuntimeExecutor()

    # ========================================================
    # STATIC BASELINE
    # ========================================================

    print("\n" + "-" * 80)
    print("STATIC BASELINE")
    print("-" * 80)

    baseline = run_baseline_trials(
        graph,
        partitions,
        executor
    )

    baseline_mean_time = mean(
        baseline["times"]
    )

    baseline_std_time = std(
        baseline["times"]
    )

    baseline_mean_throughput = mean(
        baseline["throughputs"]
    )

    print_statistics(
        "BASELINE STATISTICS",
        baseline["times"],
        baseline["throughputs"]
    )

    # ========================================================
    # LOAD CONTROLLER ONCE
    # ========================================================

    print("\n" + "-" * 80)
    print("GNN-ADAPTGRAPH CONTROLLER")
    print("-" * 80)

    print(
        "\nLoading GraphSAGE controller..."
    )

    pipeline = RuntimePipeline(
        CHECKPOINT
    )

    # ========================================================
    # CONTROLLER INFERENCE
    # ========================================================

    controller = run_controller_trials(
        pipeline,
        graph
    )

    controller_mean_time = mean(
        controller["times"]
    )

    controller_std_time = std(
        controller["times"]
    )

    controller_actions = (
        controller["actions"]
    )

    print(
        "\nPredicted actions:"
    )

    print(
        controller_actions
    )

    print(
        f"\nMean controller inference : "
        f"{controller_mean_time:.6f} sec"
    )

    print(
        f"Controller inference std  : "
        f"{controller_std_time:.6f} sec"
    )

    # ========================================================
    # ACTION CONVERSION
    # ========================================================

    adaptive_actions = convert_actions(
        controller_actions
    )

    # ========================================================
    # ADAPTIVE EXECUTION
    # ========================================================

    print("\n" + "-" * 80)
    print("GNN-ADAPTGRAPH EXECUTION")
    print("-" * 80)

    adaptive = run_adaptive_trials(
        graph,
        partitions,
        adaptive_actions,
        executor
    )

    adaptive_mean_time = mean(
        adaptive["times"]
    )

    adaptive_std_time = std(
        adaptive["times"]
    )

    adaptive_mean_throughput = mean(
        adaptive["throughputs"]
    )

    print_statistics(
        "ADAPTIVE EXECUTION STATISTICS",
        adaptive["times"],
        adaptive["throughputs"]
    )

    # ========================================================
    # ACTION DISTRIBUTION
    # ========================================================

    # Run once to obtain action labels from executor.
    action_result = execute_partitions(
        graph,
        partitions,
        adaptive_actions,
        executor
    )

    print_action_distribution(
        action_result["results"],
        "GNN-ADAPTGRAPH ACTION DISTRIBUTION"
    )

    counts = get_action_distribution(
        action_result["results"]
    )

    # ========================================================
    # EDGE VALIDATION
    # ========================================================

    baseline_edges = baseline["edges"][0]
    adaptive_edges = adaptive["edges"][0]

    baseline_valid = all(
        e == expected_edges
        for e in baseline["edges"]
    )

    adaptive_valid = all(
        e == expected_edges
        for e in adaptive["edges"]
    )

    edge_validation = (
        baseline_valid and
        adaptive_valid
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    # Runtime-only speedup
    runtime_speedup = (
        baseline_mean_time /
        adaptive_mean_time
        if adaptive_mean_time > 0
        else 0.0
    )

    # Mean end-to-end time
    end_to_end_mean_time = (
        controller_mean_time +
        adaptive_mean_time
    )

    # End-to-end speedup
    end_to_end_speedup = (
        baseline_mean_time /
        end_to_end_mean_time
        if end_to_end_mean_time > 0
        else 0.0
    )

    # Throughput gain
    throughput_gain = (
        adaptive_mean_throughput /
        baseline_mean_throughput
        if baseline_mean_throughput > 0
        else 0.0
    )

    throughput_improvement = (
        (
            adaptive_mean_throughput -
            baseline_mean_throughput
        )
        /
        baseline_mean_throughput
        *
        100.0
        if baseline_mean_throughput > 0
        else 0.0
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print(
        f"{name} FINAL CONFERENCE BENCHMARK"
    )
    print("=" * 80)

    print(
        f"\nGraph nodes                 : "
        f"{nodes}"
    )

    print(
        f"Graph edges                 : "
        f"{graph_edges}"
    )

    print(
        f"Partitions                  : "
        f"{NUM_PARTITIONS}"
    )

    print(
        f"Expected processed edges   : "
        f"{expected_edges}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print("\nSTATIC BASELINE")
    print("-" * 45)

    print(
        f"Mean execution time        : "
        f"{baseline_mean_time:.6f} sec"
    )

    print(
        f"Std. deviation             : "
        f"{baseline_std_time:.6f} sec"
    )

    print(
        f"Mean throughput             : "
        f"{baseline_mean_throughput:.2f} edges/sec"
    )

    print(
        f"Processed edges             : "
        f"{baseline_edges}"
    )

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    print("\nGRAPH SAGE CONTROLLER")
    print("-" * 45)

    print(
        f"Mean inference time         : "
        f"{controller_mean_time:.6f} sec"
    )

    print(
        f"Inference std. deviation    : "
        f"{controller_std_time:.6f} sec"
    )

    print(
        f"Predicted actions            : "
        f"{controller_actions}"
    )

    # --------------------------------------------------------
    # Adaptive
    # --------------------------------------------------------

    print("\nGNN-ADAPTGRAPH")
    print("-" * 45)

    print(
        f"Mean execution time         : "
        f"{adaptive_mean_time:.6f} sec"
    )

    print(
        f"Std. deviation              : "
        f"{adaptive_std_time:.6f} sec"
    )

    print(
        f"Mean throughput              : "
        f"{adaptive_mean_throughput:.2f} edges/sec"
    )

    print(
        f"Processed edges              : "
        f"{adaptive_edges}"
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    print("\nPERFORMANCE")
    print("-" * 45)

    print(
        f"Runtime-only speedup        : "
        f"{runtime_speedup:.4f}x"
    )

    print(
        f"End-to-end adaptive time    : "
        f"{end_to_end_mean_time:.6f} sec"
    )

    print(
        f"End-to-end speedup          : "
        f"{end_to_end_speedup:.4f}x"
    )

    print(
        f"Throughput gain             : "
        f"{throughput_gain:.4f}x"
    )

    print(
        f"Throughput improvement      : "
        f"{throughput_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nEDGE VALIDATION")
    print("-" * 45)

    print(
        f"Baseline accounting         : "
        f"{'VALID' if baseline_valid else 'INVALID'}"
    )

    print(
        f"GNN accounting              : "
        f"{'VALID' if adaptive_valid else 'INVALID'}"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("\nINTERPRETATION")
    print("-" * 45)

    if runtime_speedup > 1.0:

        print(
            "Adaptive runtime execution "
            "outperformed the static baseline."
        )

    elif runtime_speedup < 1.0:

        print(
            "Adaptive runtime execution "
            "was slower than the static baseline."
        )

    else:

        print(
            "Adaptive runtime execution "
            "matched the static baseline."
        )

    if end_to_end_speedup > 1.0:

        print(
            "The complete adaptive pipeline "
            "also outperformed the baseline."
        )

    else:

        print(
            "Controller inference overhead "
            "makes the complete end-to-end "
            "pipeline slower."
        )

    print(
        "\nNOTE:"
    )

    print(
        "THREAD_ADJUST is currently treated "
        "as a runtime configuration label "
        "because RuntimeExecutor uses "
        "Python traversal rather than true "
        "parallel execution."
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_result_row({

        "dataset": name,
        "nodes": nodes,
        "graph_edges": graph_edges,
        "partitions": NUM_PARTITIONS,

        "baseline_mean_time":
            baseline_mean_time,

        "baseline_std_time":
            baseline_std_time,

        "baseline_mean_throughput":
            baseline_mean_throughput,

        "controller_mean_time":
            controller_mean_time,

        "controller_std_time":
            controller_std_time,

        "adaptive_mean_time":
            adaptive_mean_time,

        "adaptive_std_time":
            adaptive_std_time,

        "adaptive_mean_throughput":
            adaptive_mean_throughput,

        "end_to_end_mean_time":
            end_to_end_mean_time,

        "runtime_speedup":
            runtime_speedup,

        "end_to_end_speedup":
            end_to_end_speedup,

        "throughput_gain":
            throughput_gain,

        "throughput_improvement_percent":
            throughput_improvement,

        "processed_edges":
            adaptive_edges,

        "expected_edges":
            expected_edges,

        "edge_validation":
            "VALID"
            if edge_validation
            else "INVALID",

        "no_action":
            counts["NO_ACTION"],

        "thread_adjust":
            counts["THREAD_ADJUST"],

        "rebalance":
            counts["REBALANCE"]
    })

    print(
        f"\nResults saved to: "
        f"{RESULTS_FILE}"
    )

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    # Remove previous result file so that this run
    # creates a clean conference benchmark table.

    if os.path.exists(
        RESULTS_FILE
    ):

        os.remove(
            RESULTS_FILE
        )

    print("\n" + "=" * 80)
    print(
        "GNN-ADAPTGRAPH REPEATED "
        "CONFERENCE BENCHMARK"
    )
    print("=" * 80)

    print(
        f"\nWarm-up runs     : {WARMUP_RUNS}"
    )

    print(
        f"Measured runs    : {MEASUREMENT_RUNS}"
    )

    print(
        f"Partitions       : {NUM_PARTITIONS}"
    )

    print(
        f"Checkpoint       : {CHECKPOINT}"
    )

    # ========================================================
    # DBLP
    # ========================================================

    run_dataset(
        "DBLP",
        load_dblp
    )

    # ========================================================
    # CORA
    # ========================================================

    run_dataset(
        "Cora",
        load_cora
    )

    # ========================================================
    # PUBMED
    # ========================================================

    run_dataset(
        "PubMed",
        load_pubmed
    )

    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "REPEATED CONFERENCE BENCHMARK COMPLETED"
    )
    print("=" * 80)

    print(
        f"\nFinal CSV:"
    )

    print(
        os.path.abspath(
            RESULTS_FILE
        )
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()