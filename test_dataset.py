from data.load_graph import GraphLoader
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from gnn.dataset import GraphDataset

G = GraphLoader.load_demo_graph()

partitions = PartitionManager.partition_graph(G, 4)

features = FeatureExtractor.extract_partition_features(
    G,
    partitions
)

data = GraphDataset.build_graph(features)

print("\nNode Features")
print(data.x)

print("\nEdge Index")
print(data.edge_index)

print("\nEdge Weights")
print(data.edge_attr)

print("\nNodes:", data.num_nodes)
print("Edges:", data.num_edges)