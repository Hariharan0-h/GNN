"""
PubMed Dataset Loader
"""

import networkx as nx

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


def load_pubmed():

    dataset = Planetoid(
        root="data/PubMed",
        name="PubMed"
    )

    data = dataset[0]

    graph = to_networkx(
        data,
        to_undirected=True
    )

    print("=" * 60)
    print("PubMed Citation Graph")
    print("=" * 60)

    print(
        "Nodes   :",
        graph.number_of_nodes()
    )

    print(
        "Edges   :",
        graph.number_of_edges()
    )

    print(
        "Features:",
        data.num_node_features
    )

    print(
        "Classes :",
        dataset.num_classes
    )

    print("=" * 60)

    return graph