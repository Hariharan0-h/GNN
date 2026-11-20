from gnn.datasets.dblp import load_dblp

from graph.partition_manager import PartitionManager
from graph.feature_extractor import FeatureExtractor

graph = load_dblp()

partitioner = PartitionManager(4)

partitions = partitioner.partition(graph)

extractor = FeatureExtractor()

features = extractor.extract(
    graph,
    partitions
)

print("=" * 60)
print("Runtime Features")
print("=" * 60)

print(features)

print()

print("Shape:", features.shape)