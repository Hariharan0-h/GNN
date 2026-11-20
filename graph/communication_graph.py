"""
Runtime Partition Communication Graph.

Each partition is represented as a node.
An edge represents communication between two partitions.
"""

import torch
from torch_geometric.data import Data


class CommunicationGraph:

    @staticmethod
    def build(
        original_graph,
        partitions,
        features
    ):

        # -----------------------------------------
        # Convert features to PyTorch tensor
        # -----------------------------------------

        if not isinstance(features, torch.Tensor):

            features = torch.tensor(
                features,
                dtype=torch.float
            )

        else:

            features = features.float()

        # -----------------------------------------
        # Map original graph node -> partition ID
        # -----------------------------------------

        node_partition = {}

        for pid, nodes in enumerate(partitions):

            for node in nodes:

                node_partition[node] = pid

        # -----------------------------------------
        # Count inter-partition communication
        # -----------------------------------------

        edge_weights = {}

        for u, v in original_graph.edges():

            pu = node_partition[u]
            pv = node_partition[v]

            # Ignore intra-partition edges
            if pu == pv:
                continue

            key = tuple(
                sorted(
                    (pu, pv)
                )
            )

            edge_weights[key] = (
                edge_weights.get(key, 0) + 1
            )

        # -----------------------------------------
        # Build bidirectional communication edges
        # -----------------------------------------

        edge_index = []
        edge_attr = []

        for (u, v), weight in edge_weights.items():

            edge_index.append(
                [u, v]
            )

            edge_index.append(
                [v, u]
            )

            edge_attr.append(
                float(weight)
            )

            edge_attr.append(
                float(weight)
            )

        # -----------------------------------------
        # Handle graph with no communication edges
        # -----------------------------------------

        if not edge_index:

            edge_index_tensor = torch.empty(
                (2, 0),
                dtype=torch.long
            )

            edge_attr_tensor = torch.empty(
                (0,),
                dtype=torch.float
            )

        else:

            edge_index_tensor = torch.tensor(
                edge_index,
                dtype=torch.long
            ).t().contiguous()

            edge_attr_tensor = torch.tensor(
                edge_attr,
                dtype=torch.float
            )

        # -----------------------------------------
        # PyTorch Geometric Data object
        # -----------------------------------------

        return Data(
            x=features,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor
        )