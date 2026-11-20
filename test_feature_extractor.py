from data.load_graph import GraphLoader
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor

# Load graph
G = GraphLoader.load_demo_graph()

# Partition graph
partitions = PartitionManager.partition_graph(G, k=4)

# Extract features
features = FeatureExtractor.extract_partition_features(G, partitions)

print("\nFeature Matrix")
print("-" * 50)
print(features)

print("\nShape:", features.shape)