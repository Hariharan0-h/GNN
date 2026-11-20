from gnn.graph_loader import GraphLoader
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.partition_graph import PartitionCommunicationGraph


print("=" * 60)
print("DBLP GNN-AdaptGraph Pipeline Test")
print("=" * 60)


# --------------------------------------------------
# 1. Load DBLP
# --------------------------------------------------

print("\n[1] Loading DBLP...")

graph = GraphLoader.load_dblp()

print("Nodes :", graph.number_of_nodes())
print("Edges :", graph.number_of_edges())


# --------------------------------------------------
# 2. Partition graph
# --------------------------------------------------

print("\n[2] Partitioning DBLP...")

partitions = PartitionManager.partition_graph(
    graph,
    k=4
)

print("Number of partitions:", len(partitions))

for i, partition in enumerate(partitions):

    print(
        f"Partition {i + 1}: "
        f"{len(partition)} nodes"
    )


# --------------------------------------------------
# 3. Extract 8 runtime features
# --------------------------------------------------

print("\n[3] Extracting partition features...")

features = FeatureExtractor.extract_partition_features(
    graph,
    partitions
)

print(
    "Feature matrix shape:",
    features.shape
)

print("\nFeature matrix:")
print(features)


# --------------------------------------------------
# 4. Build communication graph
# --------------------------------------------------

print("\n[4] Building communication graph...")

edge_index, edge_attr = (
    PartitionCommunicationGraph.build(
        graph,
        partitions
    )
)

print(
    "Communication edge_index shape:",
    edge_index.shape
)

print(
    "Communication edge attributes:",
    edge_attr
)


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DBLP Pipeline Test Completed")
print("=" * 60)

print(
    "DBLP nodes           :",
    graph.number_of_nodes()
)

print(
    "DBLP edges           :",
    graph.number_of_edges()
)

print(
    "Partitions           :",
    len(partitions)
)

print(
    "Feature dimensions   :",
    features.shape[1]
)

print(
    "Communication edges  :",
    edge_index.shape[1]
)

print("=" * 60)