"""
Adaptive Cost-Aware Graph Repartitioning (ACGR)

ACGR refines an existing graph partitioning by minimizing a
multi-objective cost consisting of:

1. Edge-cut cost
2. Boundary communication cost
3. Load imbalance
4. Degree skew

The algorithm performs greedy boundary-node migration.
A migration is accepted only when it strictly reduces the
overall normalized objective.
"""

import networkx as nx
import numpy as np


class ACGR:

    def __init__(
        self,
        num_partitions=4,
        alpha=0.45,
        beta=0.35,
        gamma=0.15,
        delta=0.05,
        max_iterations=5
    ):

        self.num_partitions = num_partitions

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.max_iterations = max_iterations

    # ---------------------------------------------------------
    # Edge-cut
    # ---------------------------------------------------------

    def edge_cut(self, graph, partitions):

        node_to_partition = {}

        for pid, partition in enumerate(partitions):

            for node in partition:
                node_to_partition[node] = pid

        cut = 0

        for u, v in graph.edges():

            if node_to_partition[u] != node_to_partition[v]:
                cut += 1

        return cut

    # ---------------------------------------------------------
    # Boundary communication
    # ---------------------------------------------------------

    def boundary_cost(self, graph, partitions):

        total_boundary = 0

        for partition in partitions:

            node_set = set(partition)

            for node in partition:

                for neighbor in graph.neighbors(node):

                    if neighbor not in node_set:
                        total_boundary += 1

        return total_boundary

    # ---------------------------------------------------------
    # Load imbalance
    # ---------------------------------------------------------

    def imbalance(self, partitions):

        sizes = np.array(
            [len(p) for p in partitions],
            dtype=np.float32
        )

        if len(sizes) == 0:
            return 0.0

        mean_size = np.mean(sizes)

        if mean_size == 0:
            return 0.0

        return float(
            np.std(sizes) / mean_size
        )

    # ---------------------------------------------------------
    # Degree skew
    # ---------------------------------------------------------

    def degree_skew(self, graph, partitions):

        values = []

        for partition in partitions:

            if len(partition) == 0:
                continue

            degrees = np.array(
                [
                    graph.degree(node)
                    for node in partition
                ],
                dtype=np.float32
            )

            if len(degrees) <= 1:
                values.append(0.0)
                continue

            mean = np.mean(degrees)
            std = np.std(degrees)

            if std == 0:
                values.append(0.0)
                continue

            skew = np.mean(
                ((degrees - mean) / std) ** 3
            )

            values.append(
                abs(float(skew))
            )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # ---------------------------------------------------------
    # Total objective
    # ---------------------------------------------------------

    def cost(self, graph, partitions):

        cut = self.edge_cut(
            graph,
            partitions
        )

        boundary = self.boundary_cost(
            graph,
            partitions
        )

        imbalance = self.imbalance(
            partitions
        )

        skew = self.degree_skew(
            graph,
            partitions
        )

        total_edges = max(
            graph.number_of_edges(),
            1
        )

        normalized_cut = (
            cut / total_edges
        )

        normalized_boundary = (
            boundary / (2.0 * total_edges)
        )

        normalized_skew = (
            min(skew, 10.0) / 10.0
        )

        total_cost = (

            self.alpha *
            normalized_cut

            +

            self.beta *
            normalized_boundary

            +

            self.gamma *
            imbalance

            +

            self.delta *
            normalized_skew
        )

        return float(total_cost)

    # ---------------------------------------------------------
    # Boundary nodes
    # ---------------------------------------------------------

    def get_boundary_nodes(
        self,
        graph,
        partitions
    ):

        node_to_partition = {}

        for pid, partition in enumerate(partitions):

            for node in partition:
                node_to_partition[node] = pid

        boundary_nodes = []

        for node in graph.nodes():

            source = node_to_partition[node]

            for neighbor in graph.neighbors(node):

                if (
                    node_to_partition[neighbor]
                    != source
                ):

                    boundary_nodes.append(node)
                    break

        return boundary_nodes, node_to_partition

    # ---------------------------------------------------------
    # Refinement
    # ---------------------------------------------------------

    def refine(
        self,
        graph,
        partitions
    ):

        # IMPORTANT:
        # Always make a flat list of sets.
        refined = [
            set(partition)
            for partition in partitions
        ]

        # Ensure expected number of partitions
        if len(refined) != self.num_partitions:

            raise ValueError(
                "ACGR received "
                f"{len(refined)} partitions, "
                f"expected {self.num_partitions}."
            )

        best_cost = self.cost(
            graph,
            refined
        )

        for iteration in range(
            self.max_iterations
        ):

            improved = False

            boundary_nodes, node_to_partition = (
                self.get_boundary_nodes(
                    graph,
                    refined
                )
            )

            # Try moving every boundary node
            for node in boundary_nodes:

                source = node_to_partition[node]

                # Never empty a partition
                if len(refined[source]) <= 1:
                    continue

                # Find neighbouring partitions
                candidate_targets = set()

                for neighbor in graph.neighbors(node):

                    target = node_to_partition[neighbor]

                    if target != source:
                        candidate_targets.add(target)

                for target in candidate_targets:

                    # Move node
                    refined[source].remove(node)
                    refined[target].add(node)

                    new_cost = self.cost(
                        graph,
                        refined
                    )

                    # Accept only improvement
                    if new_cost < best_cost:

                        best_cost = new_cost
                        node_to_partition[node] = target
                        improved = True

                        break

                    # Undo move
                    refined[target].remove(node)
                    refined[source].add(node)

            if not improved:
                break

        # Final safety check
        refined = [
            set(partition)
            for partition in refined
        ]

        return refined, best_cost