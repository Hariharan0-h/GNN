from data.synthetic import SyntheticGraphGenerator
from gnn.graph_dataset import GraphDatasetBuilder

G = SyntheticGraphGenerator.generate()

data = GraphDatasetBuilder.build(
    G,
    num_partitions=4
)

print("=" * 60)
print(data)

print("=" * 60)
print("Node Features")
print(data.x)

print("=" * 60)
print("Edge Index")
print(data.edge_index)

print("=" * 60)
print("Edge Weights")
print(data.edge_attr)

print("=" * 60)
print("Labels")
print(data.y)