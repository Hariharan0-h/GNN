import time
import torch

from gnn.datasets.dblp import load_dblp
from gnn.datasets.cora import load_cora
from gnn.datasets.pubmed import load_pubmed

from runtime.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from runtime.partition_graph import PartitionCommunicationGraph

from gnn.model import GraphSAGEModel


CHECKPOINT = "gnn/checkpoints/best_graphsage_model.pth"


def benchmark_dataset(name, loader):

    print("\n" + "=" * 70)
    print(f"{name} RUNTIME PERFORMANCE")
    print("=" * 70)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("\nLoading dataset...")

    t_total_start = time.perf_counter()

    graph = loader()

    print(f"Nodes : {graph.number_of_nodes()}")
    print(f"Edges : {graph.number_of_edges()}")

    # --------------------------------------------------
    # 1. Partitioning
    # --------------------------------------------------

    t0 = time.perf_counter()

    partitions = PartitionManager.partition_graph(
        graph,
        4
    )

    partition_time = time.perf_counter() - t0

    print(
        f"Partitioning Time        : "
        f"{partition_time:.6f} sec"
    )

    # --------------------------------------------------
    # 2. Feature extraction
    # --------------------------------------------------

    t0 = time.perf_counter()

    features = FeatureExtractor.extract_partition_features(
        graph,
        partitions
    )

    feature_time = time.perf_counter() - t0

    print(
        f"Feature Extraction Time  : "
        f"{feature_time:.6f} sec"
    )

    # --------------------------------------------------
    # 3. Communication graph
    # --------------------------------------------------

    t0 = time.perf_counter()

    edge_index, edge_attr = (
        PartitionCommunicationGraph.build(
            graph,
            partitions
        )
    )

    communication_time = time.perf_counter() - t0

    print(
        f"Communication Graph Time : "
        f"{communication_time:.6f} sec"
    )

    # --------------------------------------------------
    # 4. GraphSAGE inference
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

    # Warm-up inference
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

    inference_time = time.perf_counter() - t0

    predictions = torch.argmax(
        output,
        dim=1
    )

    # --------------------------------------------------
    # Total pipeline
    # --------------------------------------------------

    total_time = (
        partition_time
        + feature_time
        + communication_time
        + inference_time
    )

    print(
        f"GraphSAGE Inference Time : "
        f"{inference_time:.6f} sec"
    )

    print(
        f"Total Pipeline Time      : "
        f"{total_time:.6f} sec"
    )

    # --------------------------------------------------
    # Actions
    # --------------------------------------------------

    actions = {
        0: "NO_ACTION",
        1: "THREAD_ADJUST",
        2: "REBALANCE"
    }

    print("\nPredicted Actions:")

    for i, prediction in enumerate(predictions):

        print(
            f"Partition {i + 1}: "
            f"{actions[int(prediction)]}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nSummary")
    print("-" * 70)
    print(f"Dataset                 : {name}")
    print(f"Nodes                   : {graph.number_of_nodes()}")
    print(f"Edges                   : {graph.number_of_edges()}")
    print(f"Partitions              : {len(partitions)}")
    print(f"Features                : {features.shape[1]}")
    print(f"Communication edges     : {edge_index.shape[1]}")
    print(f"Partitioning time       : {partition_time:.6f} s")
    print(f"Feature extraction time : {feature_time:.6f} s")
    print(f"Communication time      : {communication_time:.6f} s")
    print(f"GNN inference time      : {inference_time:.6f} s")
    print(f"Total pipeline time     : {total_time:.6f} s")


print("\n" + "=" * 70)
print("GNN-ADAPTGRAPH RUNTIME BENCHMARK")
print("=" * 70)

benchmark_dataset(
    "DBLP",
    load_dblp
)

benchmark_dataset(
    "Cora",
    load_cora
)

benchmark_dataset(
    "PubMed",
    load_pubmed
)

print("\n" + "=" * 70)
print("RUNTIME BENCHMARK COMPLETED")
print("=" * 70)