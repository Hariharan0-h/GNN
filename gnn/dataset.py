import numpy as np
import torch
from torch_geometric.data import Data


class GraphDataset:

    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (
            np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
        )

    @staticmethod
    def build_graph(feature_matrix):

        num_nodes = feature_matrix.shape[0]

        edge_index = []
        edge_weight = []

        # Fully connect partitions with similarity weights
        for i in range(num_nodes):
            for j in range(num_nodes):

                if i == j:
                    continue

                sim = GraphDataset.cosine_similarity(
                    feature_matrix[i],
                    feature_matrix[j]
                )

                edge_index.append([i, j])
                edge_weight.append(sim)

        edge_index = torch.tensor(edge_index).t().contiguous()

        x = torch.tensor(feature_matrix, dtype=torch.float)

        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight
        )