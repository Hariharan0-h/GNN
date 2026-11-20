from gnn.datasets.dblp import load_dblp

from graph.partition_manager import PartitionManager
from graph.feature_extractor import FeatureExtractor
from graph.graph_builder import GraphBuilder

print("=" * 60)
print("Testing Graph Builder")
print("=" * 60)

graph = load_dblp()

partitions = PartitionManager(4).partition(graph)

features = FeatureExtractor().extract(
    graph,
    partitions
)

runtime_graph = GraphBuilder().build(features)

print(runtime_graph)

print()

print("Node Features :", runtime_graph.x.shape)
print("Edges         :", runtime_graph.edge_index.shape)