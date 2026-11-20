"""
Runtime Feature Extractor

Extracts 8 topology and partition features
for every graph partition.
"""

import networkx as nx
import numpy as np


class FeatureExtractor:

    @staticmethod
    def extract_partition_features(G, partitions):
        """
        Extract 8 features for every partition.

        Features
        --------
        1. Degree Skewness
        2. Load Imbalance Ratio
        3. Average Clustering Coefficient
        4. Edge Density
        5. Average Degree
        6. Partition Size Ratio
        7. Boundary Edge Count
        8. Edge-Cut Ratio
        """

        features = []

        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()

        for nodes in partitions:

            node_set = set(nodes)

            subgraph = G.subgraph(nodes)

            # --------------------------------
            # Degree statistics
            # --------------------------------

            degrees = np.array(
                [d for _, d in subgraph.degree()],
                dtype=np.float32
            )

            # 1. Degree Skewness
            if (
                len(degrees) <= 1
                or np.std(degrees) == 0
            ):

                degree_skew = 0.0

            else:

                mean = np.mean(degrees)
                std = np.std(degrees)

                degree_skew = np.mean(
                    ((degrees - mean) / std) ** 3
                )

            # 2. Load Imbalance Ratio
            if total_nodes > 0:

                imbalance_ratio = (
                    len(nodes) / total_nodes
                )

            else:

                imbalance_ratio = 0.0

            # 3. Average Clustering Coefficient
            if subgraph.number_of_nodes() > 1:

                clustering = (
                    nx.average_clustering(
                        subgraph
                    )
                )

            else:

                clustering = 0.0

            # 4. Edge Density
            if subgraph.number_of_nodes() > 1:

                density = nx.density(
                    subgraph
                )

            else:

                density = 0.0

            # 5. Average Degree
            if len(degrees) > 0:

                avg_degree = float(
                    np.mean(degrees)
                )

            else:

                avg_degree = 0.0

            # 6. Partition Size Ratio
            if total_nodes > 0:

                partition_size_ratio = (
                    len(nodes) / total_nodes
                )

            else:

                partition_size_ratio = 0.0

            # 7. Boundary Edge Count
            boundary_edges = 0

            for u in nodes:

                for v in G.neighbors(u):

                    if v not in node_set:

                        boundary_edges += 1

            # 8. Edge-Cut Ratio
            if total_edges > 0:

                edge_cut_ratio = (
                    boundary_edges / total_edges
                )

            else:

                edge_cut_ratio = 0.0

            features.append([
                degree_skew,
                imbalance_ratio,
                clustering,
                density,
                avg_degree,
                partition_size_ratio,
                boundary_edges,
                edge_cut_ratio
            ])

        return np.array(
            features,
            dtype=np.float32
        )