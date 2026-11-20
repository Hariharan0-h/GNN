import time
from collections import Counter

import torch

from gnn.datasets.dblp import load_dblp
from gnn.datasets.cora import load_cora
from gnn.datasets.pubmed import load_pubmed

from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.partition_graph import PartitionCommunicationGraph
from runtime.decision_engine import RuntimeDecisionEngine

from gnn.model import GraphSAGEModel


CHECKPOINT = "gnn/checkpoints/best_graphsage_model.pth"


def run_experiment(name, loader):

    print("\n" + "=" * 70)
    print(f"{name} BASELINE COMPARISON")
    print("=" * 70)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    graph = loader()

    print(f"Nodes : {graph.number_of_nodes()}")
    print(f"Edges : {graph.number_of_edges()}")

    # --------------------------------------------------
    # Static baseline
    # --------------------------------------------------

    t0 = time.perf_counter()

    partitions = PartitionManager.partition_graph(
        graph,
        4
    )

    static_time = time.perf_counter() - t0

    print("\nStatic Partitioning")
    print(f"Time : {static_time:.6f} sec")

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

    t0 = time.perf_counter()

    features = FeatureExtractor.extract_partition_features(
        graph,
        partitions
    )

    feature_time = time.perf_counter() - t0

    # --------------------------------------------------
    # Rule-based controller
    # --------------------------------------------------

    t0 = time.perf_counter()

    rule_actions = [
        RuntimeDecisionEngine.predict(feature)
        for feature in features
    ]

    rule_time = time.perf_counter() - t0

    # --------------------------------------------------
    # Communication graph
    # --------------------------------------------------

    t0 = time.perf_counter()

    edge_index, edge_attr = (
        PartitionCommunicationGraph.build(
            graph,
            partitions
        )
    )

    communication_time = time.perf_counter() - t0

    # --------------------------------------------------
    # GNN controller
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = GraphSAGEModel().to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device
        )
    )

    model.eval()

    x = torch.tensor(
        features,
        dtype=torch.float
    ).to(device)

    edge_index = edge_index.to(device)

    # Warm-up
    with torch.no_grad():
        model(x, edge_index)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        output = model(
            x,
            edge_index
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    gnn_time = time.perf_counter() - t0

    gnn_actions = torch.argmax(
        output,
        dim=1
    ).cpu().tolist()

    # --------------------------------------------------
    # Action names
    # --------------------------------------------------

    action_names = {
        0: "NO_ACTION",
        1: "THREAD_ADJUST",
        2: "REBALANCE"
    }

    rule_names = [
        action_names[a]
        for a in rule_actions
    ]

    gnn_names = [
        action_names[a]
        for a in gnn_actions
    ]

    # --------------------------------------------------
    # Action distributions
    # --------------------------------------------------

    rule_distribution = Counter(rule_names)
    gnn_distribution = Counter(gnn_names)

    print("\nRule-based Actions")
    print("-" * 40)

    for action in action_names.values():

        print(
            f"{action:16s}: "
            f"{rule_distribution.get(action, 0)}"
        )

    print("\nGNN-AdaptGraph Actions")
    print("-" * 40)

    for action in action_names.values():

        print(
            f"{action:16s}: "
            f"{gnn_distribution.get(action, 0)}"
        )

    # --------------------------------------------------
    # Decision agreement
    # --------------------------------------------------

    agreement = sum(
        r == g
        for r, g in zip(
            rule_actions,
            gnn_actions
        )
    )

    agreement_rate = (
        agreement / len(rule_actions)
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nTiming Comparison")
    print("-" * 70)

    print(
        f"Static partitioning       : "
        f"{static_time:.6f} sec"
    )

    print(
        f"Feature extraction        : "
        f"{feature_time:.6f} sec"
    )

    print(
        f"Rule-based decision       : "
        f"{rule_time:.6f} sec"
    )

    print(
        f"Communication graph       : "
        f"{communication_time:.6f} sec"
    )

    print(
        f"GNN inference             : "
        f"{gnn_time:.6f} sec"
    )

    print(
        f"Rule/GNN agreement        : "
        f"{agreement_rate * 100:.2f}%"
    )

    print("\nDataset completed.")

    return {
        "dataset": name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "static_time": static_time,
        "feature_time": feature_time,
        "rule_time": rule_time,
        "communication_time": communication_time,
        "gnn_time": gnn_time,
        "rule_actions": rule_names,
        "gnn_actions": gnn_names,
        "agreement": agreement_rate
    }


print("\n" + "=" * 70)
print("GNN-ADAPTGRAPH BASELINE COMPARISON")
print("=" * 70)

results = []

results.append(
    run_experiment(
        "DBLP",
        load_dblp
    )
)

results.append(
    run_experiment(
        "Cora",
        load_cora
    )
)

results.append(
    run_experiment(
        "PubMed",
        load_pubmed
    )
)

print("\n" + "=" * 70)
print("FINAL BASELINE SUMMARY")
print("=" * 70)

for result in results:

    print(
        f"\n{result['dataset']}"
    )

    print(
        f"Static partitioning : "
        f"{result['static_time']:.6f} s"
    )

    print(
        f"Rule decision       : "
        f"{result['rule_time']:.6f} s"
    )

    print(
        f"GNN inference       : "
        f"{result['gnn_time']:.6f} s"
    )

    print(
        f"Agreement            : "
        f"{result['agreement'] * 100:.2f}%"
    )

print("\nBaseline experiment completed.")