"""
Graph Dataset Generator

Generates many PyTorch Geometric Data objects
for GraphSAGE training.
"""

from tqdm import tqdm

from data.synthetic import SyntheticGraphGenerator
from gnn.graph_dataset import GraphDatasetBuilder


class GraphTrainingDataset:

    @staticmethod
    def build(
        num_graphs=5000,
        partitions=4
    ):

        dataset = []

        for _ in tqdm(range(num_graphs), desc="Generating Graph Dataset"):

            G = SyntheticGraphGenerator.generate()

            data = GraphDatasetBuilder.build(
                G,
                num_partitions=partitions
            )

            dataset.append(data)

        return dataset