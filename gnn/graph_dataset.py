"""
Graph Dataset Builder

Creates PyTorch Geometric Data objects
for GraphSAGE training.
"""

import torch
from torch_geometric.data import Data

from runtime.partition_manager import PartitionManager
from runtime.partition_graph import PartitionCommunicationGraph
from monitor.feature_extractor import FeatureExtractor
from runtime.decision_engine import RuntimeDecisionEngine


class GraphDatasetBuilder:

    @staticmethod
    def build(G, num_partitions=4):

        # Partition graph
        partitions = PartitionManager.partition_graph(
            G,
            num_partitions
        )

        # Extract partition features
        features = FeatureExtractor.extract_partition_features(
            G,
            partitions
        )

        # Node features
        x = torch.tensor(
            features,
            dtype=torch.float
        )

        # Labels
        labels = [
            RuntimeDecisionEngine.predict(f)
            for f in features
        ]

        y = torch.tensor(
            labels,
            dtype=torch.long
        )

        # Communication graph
        edge_index, edge_attr = (
            PartitionCommunicationGraph.build(
                G,
                partitions
            )
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        return data