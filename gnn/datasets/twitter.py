"""
ego-Twitter Dataset Loader

Loads Twitter ego network dataset
using PyTorch Geometric.
"""

from torch_geometric.datasets import TUDataset


def load_twitter():

    dataset = TUDataset(
        root="data/Twitter",
        name="TWITTER"
    )


    print("=" * 60)
    print("ego-Twitter Dataset")
    print("=" * 60)

    print(
        "Graphs:",
        len(dataset)
    )


    data = dataset[0]

    print(data)

    print(
        "Nodes:",
        data.num_nodes
    )

    print(
        "Edges:",
        data.num_edges
    )


    return dataset