"""
Dataset Generator

Generates training samples for GraphSAGE.
"""

import random
import numpy as np
import pandas as pd

from data.synthetic import SyntheticGraphGenerator
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.decision_engine import RuntimeDecisionEngine


class DatasetGenerator:

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def generate(self,
                 num_graphs=1000,
                 partitions=4):

        dataset = []

        for i in range(num_graphs):

            G = SyntheticGraphGenerator.generate()

            # Skip disconnected graphs
            if G.number_of_nodes() < partitions:
                continue

            try:
                graph_partitions = PartitionManager.partition_graph(
                    G,
                    partitions
                )

                features = FeatureExtractor.extract_partition_features(
                    G,
                    graph_partitions
                )

                for feature in features:

                    label = RuntimeDecisionEngine.predict(feature)

                    dataset.append([
                        feature[0],
                        feature[1],
                        feature[2],
                        feature[3],
                        label
                    ])

            except Exception:
                # Ignore graphs that fail partitioning
                continue

        columns = [
            "degree_skew",
            "imbalance",
            "clustering",
            "density",
            "label"
        ]

        return pd.DataFrame(dataset, columns=columns)