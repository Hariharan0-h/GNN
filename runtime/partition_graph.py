"""
Partition Communication Graph

Creates a graph where each node is a partition and
edges represent communication between partitions.
"""

import torch


class PartitionCommunicationGraph:

    @staticmethod
    def build(original_graph, partitions):

        # Map node -> partition
        node_partition = {}

        for pid, nodes in enumerate(partitions):
            for node in nodes:
                node_partition[node] = pid

        edge_weights = {}

        # Count communication edges
        for u, v in original_graph.edges():

            pu = node_partition[u]
            pv = node_partition[v]

            if pu == pv:
                continue

            key = tuple(sorted((pu, pv)))

            edge_weights[key] = edge_weights.get(key, 0) + 1

        edge_index = []
        edge_attr = []

        for (u, v), weight in edge_weights.items():

            edge_index.append([u, v])
            edge_index.append([v, u])

            edge_attr.append(weight)
            edge_attr.append(weight)

        if len(edge_index) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float),
            )

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_index, edge_attr