from gnn.datasets.cora import load_cora

from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from graph.communication_graph import CommunicationGraph

import torch


print("=" * 60)
print("Cora GNN-AdaptGraph Pipeline Test")
print("=" * 60)


# --------------------------------------------------
# 1. Load Cora
# --------------------------------------------------

print("\n[1] Loading Cora...")

graph = load_cora()


# --------------------------------------------------
# 2. Partition graph
# --------------------------------------------------

print("\n[2] Partitioning Cora...")

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
        f"Partition {i+1}:",
        len(partition),
        "nodes"
    )


# --------------------------------------------------
# 3. Extract runtime features
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

features_tensor = torch.tensor(
    features,
    dtype=torch.float
)

communication_graph = CommunicationGraph.build(
    graph,
    partitions,
    features_tensor
)

print(
    "Communication edge_index shape:",
    communication_graph.edge_index.shape
)

print(
    "Communication edges:",
    communication_graph.edge_index.shape[1]
)

print(
    "Communication edge attributes:",
    communication_graph.edge_attr
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("Cora Pipeline Test Completed")
print("=" * 60)

print(
    "Cora nodes           :",
    graph.number_of_nodes()
)

print(
    "Cora edges           :",
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
    communication_graph.edge_index.shape[1]
)

print("=" * 60)