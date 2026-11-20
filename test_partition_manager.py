from gnn.datasets.dblp import load_dblp
from graph.partition_manager import PartitionManager

print("=" * 60)
print("Loading DBLP")
print("=" * 60)

graph = load_dblp()

partitioner = PartitionManager(num_partitions=4)

partitions = partitioner.partition(graph)

print()

print("Partitions Created")

for i, p in enumerate(partitions):
    print(f"Partition {i+1}: {len(p)} nodes")