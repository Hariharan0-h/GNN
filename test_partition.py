from data.load_graph import GraphLoader
from runtime.partition_manager import PartitionManager

# Load demo graph
G = GraphLoader.load_demo_graph()

# Partition graph
partitions = PartitionManager.partition_graph(G, k=4)

print("\nGraph Partition Summary")
print("-" * 40)

for i, part in enumerate(partitions):
    print(f"Partition {i+1}: {len(part)} nodes")