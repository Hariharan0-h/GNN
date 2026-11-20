"""
Graph Partition Manager

Partitions a NetworkX graph into balanced partitions
using recursive Kernighan-Lin bisection.
"""

from networkx.algorithms.community import (
    kernighan_lin_bisection
)


class PartitionManager:

    def __init__(self, num_partitions=4):

        self.num_partitions = num_partitions

    def partition(self, graph):

        partitions = [
            set(graph.nodes())
        ]

        while len(partitions) < self.num_partitions:

            # Select the largest current partition
            largest = max(
                partitions,
                key=len
            )

            partitions.remove(largest)

            subgraph = graph.subgraph(
                largest
            )

            # Cannot split a single-node partition
            if subgraph.number_of_nodes() <= 1:

                partitions.append(largest)
                break

            p1, p2 = kernighan_lin_bisection(
                subgraph
            )

            partitions.append(
                set(p1)
            )

            partitions.append(
                set(p2)
            )

        return partitions