from gnn.datasets.pubmed import load_pubmed

from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from graph.communication_graph import CommunicationGraph


print("=" * 60)
print("PubMed GNN-AdaptGraph Pipeline Test")
print("=" * 60)


# --------------------------------------------------
# Load PubMed
# --------------------------------------------------

print("\n[1] Loading PubMed...")

graph = load_pubmed()

print(
    "Nodes :",
    graph.number_of_nodes()
)

print(
    "Edges :",
    graph.number_of_edges()
)


# --------------------------------------------------
# Partition
# --------------------------------------------------

print("\n[2] Partitioning PubMed...")

partition_manager = PartitionManager(
    num_partitions=4
)

partitions = partition_manager.partition(
    graph
)

print(
    "Number of partitions:",
    len(partitions)
)

for i, partition in enumerate(partitions):

    print(
        f"Partition {i + 1}: "
        f"{len(partition)} nodes"
    )


# --------------------------------------------------
# Feature extraction
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
# Communication graph
# --------------------------------------------------

print("\n[4] Building communication graph...")

runtime_graph = CommunicationGraph.build(
    graph,
    partitions,
    __import__("torch").tensor(
        features,
        dtype=__import__("torch").float
    )
)

print(
    "Communication edge_index shape:",
    runtime_graph.edge_index.shape
)

print(
    "Communication edges:",
    runtime_graph.edge_index.shape[1]
)

print(
    "Communication edge attributes:",
    runtime_graph.edge_attr
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("PubMed Pipeline Test Completed")
print("=" * 60)

print(
    "PubMed nodes           :",
    graph.number_of_nodes()
)

print(
    "PubMed edges           :",
    graph.number_of_edges()
)

print(
    "Partitions             :",
    len(partitions)
)

print(
    "Feature dimensions     :",
    features.shape[1]
)

print(
    "Communication edges    :",
    runtime_graph.edge_index.shape[1]
)

print("=" * 60)