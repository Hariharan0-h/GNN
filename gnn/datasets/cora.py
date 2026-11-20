"""
Cora Dataset Loader

Loads the Cora citation network using
PyTorch Geometric and converts it into
a NetworkX graph for GNN-AdaptGraph.
"""

import networkx as nx
from torch_geometric.datasets import Planetoid


def load_cora():

    dataset = Planetoid(
        root="data/Cora",
        name="Cora"
    )

    data = dataset[0]

    # Convert PyG graph to NetworkX
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(data.num_nodes))

    # Add edges
    edges = data.edge_index.t().tolist()

    G.add_edges_from(edges)

    print("=" * 60)
    print("Cora Citation Graph")
    print("=" * 60)

    print("Nodes   :", G.number_of_nodes())
    print("Edges   :", G.number_of_edges())
    print("Features:", data.num_node_features)
    print("Classes :", dataset.num_classes)
    print("=" * 60)

    return G