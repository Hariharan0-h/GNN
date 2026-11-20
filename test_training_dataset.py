from gnn.dataset_builder import GraphTrainingDataset

dataset = GraphTrainingDataset.build(
    num_graphs=10
)

print("=" * 60)
print("Dataset Size")
print(len(dataset))

print("=" * 60)
print(dataset[0])