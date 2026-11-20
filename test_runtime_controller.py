from gnn.dataset_builder import GraphTrainingDataset
from controller.runtime_controller import RuntimeController

print("=" * 60)
print("Generating Sample Graph")
print("=" * 60)

dataset = GraphTrainingDataset.build(
    num_graphs=1
)

graph = dataset[0]

controller = RuntimeController()

actions = controller.predict(graph)

print("\nPredicted Runtime Actions\n")

for i, action in enumerate(actions):

    print(
        f"Partition {i+1}: {action}"
    )