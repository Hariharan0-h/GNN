from data.synthetic import SyntheticGraphGenerator
from gnn.graph_dataset import GraphDatasetBuilder
from gnn.model import GraphSAGEModel

# Generate one graph
G = SyntheticGraphGenerator.generate()

data = GraphDatasetBuilder.build(
    G,
    num_partitions=4
)

# Create model
model = GraphSAGEModel()

# Forward pass
output = model(
    data.x,
    data.edge_index
)

print("=" * 60)
print("Model Output")
print(output)

print("=" * 60)
print("Output Shape")
print(output.shape)