"""
Runtime Feature Extractor
"""

import torch
import networkx as nx
from torch_geometric.utils import to_networkx


class FeatureExtractor:

    def extract(self, data, partitions):

        G = to_networkx(data, to_undirected=True)

        features = []

        total_nodes = data.num_nodes

        for nodes in partitions:

            node_list = nodes.tolist()

            subgraph = G.subgraph(node_list)

            # Feature 1: Partition Size Ratio
            size_ratio = len(node_list) / total_nodes

            # Feature 2: Average Degree
            if subgraph.number_of_nodes() > 0:
                avg_degree = sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()
            else:
                avg_degree = 0.0

            # Feature 3: Edge Density
            density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0

            # Feature 4: Average Clustering Coefficient
            clustering = (
                nx.average_clustering(subgraph)
                if subgraph.number_of_nodes() > 1
                else 0.0
            )

            features.append([
                size_ratio,
                avg_degree,
                density,
                clustering
            ])

        return torch.tensor(features, dtype=torch.float)