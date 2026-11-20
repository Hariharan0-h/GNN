import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection


class PartitionManager:
    """
    Partition a graph into K balanced partitions.
    """

    @staticmethod
    def partition_graph(G, k=4):
        """
        Recursively partition the graph using Kernighan-Lin bisection.
        """

        partitions = [set(G.nodes())]

        while len(partitions) < k:
            largest = max(partitions, key=len)
            partitions.remove(largest)

            subgraph = G.subgraph(largest)

            if subgraph.number_of_nodes() <= 1:
                partitions.append(largest)
                continue

            p1, p2 = kernighan_lin_bisection(
                subgraph,
                seed=42
            )

            partitions.append(set(p1))
            partitions.append(set(p2))

        return partitions