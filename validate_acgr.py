"""
ACGR Validation

Adaptive Cost-Aware Graph Repartitioning
for GNN-AdaptGraph
"""

import numpy as np

from gnn.datasets.cora import load_cora
from graph.partition_manager import PartitionManager
from runtime.acgr import ACGR


NUM_PARTITIONS = 4


# ============================================================
# Partition statistics
# ============================================================

def partition_stats(graph, partitions):

    # --------------------------------------------------------
    # Validate and normalize partition structure
    # --------------------------------------------------------

    normalized = []

    for partition in partitions:

        if isinstance(partition, set):

            normalized.append(set(partition))

        elif isinstance(partition, (list, tuple)):

            # Make sure this is not nested
            for node in partition:

                if isinstance(
                    node,
                    (set, list, tuple)
                ):

                    raise TypeError(
                        "Nested partition detected."
                    )

            normalized.append(set(partition))

        else:

            raise TypeError(
                "Invalid partition type: "
                f"{type(partition)}"
            )

    partitions = normalized

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(partitions) != NUM_PARTITIONS:

        raise ValueError(
            f"Expected {NUM_PARTITIONS} partitions, "
            f"got {len(partitions)}"
        )

    # --------------------------------------------------------
    # Node -> partition mapping
    # --------------------------------------------------------

    node_to_partition = {}

    for pid, partition in enumerate(partitions):

        for node in partition:

            if isinstance(
                node,
                (set, list, tuple)
            ):

                raise TypeError(
                    "Partition contains a nested "
                    "set/list/tuple."
                )

            node_to_partition[node] = pid

    # --------------------------------------------------------
    # Check all graph nodes are assigned
    # --------------------------------------------------------

    missing_nodes = [
        node
        for node in graph.nodes()
        if node not in node_to_partition
    ]

    if missing_nodes:

        raise ValueError(
            f"{len(missing_nodes)} graph nodes "
            "are missing from partitions."
        )

    # --------------------------------------------------------
    # Edge cut
    # --------------------------------------------------------

    edge_cut = 0

    for u, v in graph.edges():

        if (
            node_to_partition[u]
            != node_to_partition[v]
        ):

            edge_cut += 1

    # --------------------------------------------------------
    # Boundary edges
    # --------------------------------------------------------

    boundary_edges = 0

    for partition in partitions:

        node_set = set(partition)

        for node in partition:

            for neighbor in graph.neighbors(node):

                if neighbor not in node_set:

                    boundary_edges += 1

    # --------------------------------------------------------
    # Load imbalance
    # --------------------------------------------------------

    sizes = np.array(
        [
            len(partition)
            for partition in partitions
        ],
        dtype=np.float32
    )

    if len(sizes) == 0:

        load_imbalance = 0.0

    else:

        mean_size = float(
            np.mean(sizes)
        )

        if mean_size == 0:

            load_imbalance = 0.0

        else:

            load_imbalance = float(
                np.std(sizes) / mean_size
            )

    return {
        "sizes": [
            len(partition)
            for partition in partitions
        ],
        "edge_cut": edge_cut,
        "boundary_edges": boundary_edges,
        "load_imbalance": load_imbalance
    }


# ============================================================
# Print statistics
# ============================================================

def print_stats(title, stats):

    print()
    print(title)
    print(
        "Sizes           :",
        stats["sizes"]
    )

    print(
        "Edge cut        :",
        stats["edge_cut"]
    )

    print(
        "Boundary edges  :",
        stats["boundary_edges"]
    )

    print(
        "Load imbalance  :",
        f"{stats['load_imbalance']:.6f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ACGR VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Cora
    # --------------------------------------------------------

    graph = load_cora()

    print()
    print("Graph loaded")
    print(
        "Nodes :",
        graph.number_of_nodes()
    )
    print(
        "Edges :",
        graph.number_of_edges()
    )

    # --------------------------------------------------------
    # Initial partition
    # --------------------------------------------------------

    manager = PartitionManager(
        num_partitions=NUM_PARTITIONS
    )

    partitions = manager.partition(
        graph
    )

    # Safety conversion
    partitions = [
        set(p)
        for p in partitions
    ]

    initial_stats = partition_stats(
        graph,
        partitions
    )

    print_stats(
        "Initial partitions",
        initial_stats
    )

    # --------------------------------------------------------
    # ACGR
    # --------------------------------------------------------

    acgr = ACGR(
        num_partitions=NUM_PARTITIONS,

        alpha=0.45,
        beta=0.35,
        gamma=0.15,
        delta=0.05,

        max_iterations=5
    )

    initial_cost = acgr.cost(
        graph,
        partitions
    )

    print(
        "Initial ACGR cost:",
        f"{initial_cost:.6f}"
    )

    # --------------------------------------------------------
    # Refine
    # --------------------------------------------------------

    refined_partitions, final_cost = acgr.refine(
        graph,
        partitions
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Ensure ACGR output is a flat list of sets
    # --------------------------------------------------------

    refined_partitions = [
        set(p)
        for p in refined_partitions
    ]

    # Diagnostic output
    print()
    print("ACGR output validation")
    print(
        "Type:",
        type(refined_partitions)
    )

    print(
        "Number of partitions:",
        len(refined_partitions)
    )

    print(
        "Partition types:",
        [
            type(p)
            for p in refined_partitions
        ]
    )

    print(
        "Partition sizes:",
        [
            len(p)
            for p in refined_partitions
        ]
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    refined_stats = partition_stats(
        graph,
        refined_partitions
    )

    print_stats(
        "After ACGR refinement",
        refined_stats
    )

    print(
        "Final ACGR cost :",
        f"{final_cost:.6f}"
    )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    initial_cut = initial_stats[
        "edge_cut"
    ]

    final_cut = refined_stats[
        "edge_cut"
    ]

    initial_boundary = initial_stats[
        "boundary_edges"
    ]

    final_boundary = refined_stats[
        "boundary_edges"
    ]

    initial_imbalance = initial_stats[
        "load_imbalance"
    ]

    final_imbalance = refined_stats[
        "load_imbalance"
    ]

    if initial_cut > 0:

        cut_improvement = (
            (initial_cut - final_cut)
            / initial_cut
            * 100.0
        )

    else:

        cut_improvement = 0.0

    if initial_boundary > 0:

        boundary_improvement = (
            (initial_boundary - final_boundary)
            / initial_boundary
            * 100.0
        )

    else:

        boundary_improvement = 0.0

    if initial_imbalance > 0:

        imbalance_improvement = (
            (initial_imbalance - final_imbalance)
            / initial_imbalance
            * 100.0
        )

    else:

        # Both equal zero means no imbalance
        if final_imbalance == 0:

            imbalance_improvement = 0.0

        else:

            imbalance_improvement = -100.0

    if initial_cost > 0:

        cost_improvement = (
            (initial_cost - final_cost)
            / initial_cost
            * 100.0
        )

    else:

        cost_improvement = 0.0

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ACGR IMPROVEMENT")
    print("=" * 70)

    print(
        "Edge-cut improvement   :",
        f"{cut_improvement:.2f}%"
    )

    print(
        "Boundary improvement   :",
        f"{boundary_improvement:.2f}%"
    )

    print(
        "Imbalance improvement  :",
        f"{imbalance_improvement:.2f}%"
    )

    print(
        "Cost improvement       :",
        f"{cost_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()

    if final_cost <= initial_cost:

        print("=" * 70)
        print("ACGR VALIDATION: PASS")
        print(
            "The refined partition has equal or "
            "lower objective cost."
        )
        print("=" * 70)

    else:

        print("=" * 70)
        print("ACGR VALIDATION: FAIL")
        print(
            "The refined partition increased "
            "the objective cost."
        )
        print("=" * 70)


if __name__ == "__main__":
    main()