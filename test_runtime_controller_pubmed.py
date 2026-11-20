from collections import Counter

from gnn.datasets.pubmed import load_pubmed
from controller.runtime_pipeline import RuntimePipeline


print("=" * 60)
print("PubMed Runtime Controller")
print("=" * 60)

graph = load_pubmed()

checkpoint = (
    "gnn/checkpoints/"
    "best_graphsage_model.pth"
)

pipeline = RuntimePipeline(
    checkpoint
)

actions = pipeline.process(
    graph
)

print()
print("=" * 60)
print("PubMed Predicted Runtime Actions")
print("=" * 60)

for i, action in enumerate(actions):
    print(
        f"Partition {i + 1}: {action}"
    )

counts = Counter(actions)
total = len(actions)

print()
print("Action Distribution")
print("-" * 40)

for action in [
    "NO_ACTION",
    "THREAD_ADJUST",
    "REBALANCE"
]:
    count = counts.get(action, 0)
    percentage = (count / total) * 100

    print(
        f"{action:18s}: "
        f"{count:2d} "
        f"({percentage:6.2f}%)"
    )

print()
print("Total partitions:", total)
print("=" * 60)