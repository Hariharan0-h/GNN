from data.load_graph import GraphLoader
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from gnn.preprocessing import FeaturePreprocessor

# Load graph
G = GraphLoader.load_demo_graph()

# Partition
partitions = PartitionManager.partition_graph(G, 4)

# Extract features
features = FeatureExtractor.extract_partition_features(
    G,
    partitions
)

print("Original Features")
print(features)

# Normalize
preprocessor = FeaturePreprocessor()

scaled = preprocessor.fit_transform(features)

print("\nNormalized Features")
print(scaled)