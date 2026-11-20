from gnn.datasets.dblp import load_dblp

from graph.partition_manager import PartitionManager
from graph.feature_extractor import FeatureExtractor
from graph.communication_graph import CommunicationGraph

graph = load_dblp()

partitions = PartitionManager(4).partition(graph)

features = FeatureExtractor().extract(
    graph,
    partitions
)

comm_graph = CommunicationGraph().build(
    features
)

print("=" * 60)
print("Partition Communication Graph")
print("=" * 60)

print(comm_graph)

print()

print("Node Feature Shape :", comm_graph.x.shape)
print("Edge Index Shape   :", comm_graph.edge_index.shape)