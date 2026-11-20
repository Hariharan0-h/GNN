from data.load_graph import GraphLoader
from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.decision_engine import RuntimeDecisionEngine

G = GraphLoader.load_demo_graph()

partitions = PartitionManager.partition_graph(G, 4)

features = FeatureExtractor.extract_partition_features(
    G,
    partitions
)

print("=" * 60)
print("Runtime Decisions")
print("=" * 60)

for i, feature in enumerate(features):

    label = RuntimeDecisionEngine.predict(feature)

    if label == 0:
        action = "NO ACTION"

    elif label == 1:
        action = "REBALANCE"

    else:
        action = "THREAD ADJUST"

    print(f"Partition {i+1}")
    print("Features :", feature)
    print("Decision :", action)
    print("-" * 60)