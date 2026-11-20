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


def benchmark(name, loader, k):

    graph = loader()

    t0 = time.perf_counter()

    partitions = PartitionManager.partition_graph(
        graph,
        k
    )

    partition_time = time.perf_counter() - t0

    t0 = time.perf_counter()

    features = FeatureExtractor.extract_partition_features(
        graph,
        partitions
    )

    feature_time = time.perf_counter() - t0

    t0 = time.perf_counter()

    edge_index, edge_attr = (
        PartitionCommunicationGraph.build(
            graph,
            partitions
        )
    )

    communication_time = time.perf_counter() - t0

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

    total_time = (
        partition_time
        + feature_time
        + communication_time
        + inference_time
    )

    predictions = torch.argmax(
        output,
        dim=1
    ).cpu().tolist()

    print(
        f"{name:7s} | "
        f"K={k:<2d} | "
        f"Partition={partition_time:.6f}s | "
        f"Features={feature_time:.6f}s | "
        f"Comm={communication_time:.6f}s | "
        f"GNN={inference_time:.6f}s | "
        f"Total={total_time:.6f}s"
    )

    print(
        f"         Nodes/partition: "
        f"{[len(p) for p in partitions]}"
    )

    print(
        f"         Actions: {predictions}"
    )


def run_dataset(name, loader):

    print("\n" + "=" * 90)
    print(f"{name} SCALABILITY")
    print("=" * 90)

    for k in [2, 4, 8]:

        benchmark(
            name,
            loader,
            k
        )


print("\n" + "=" * 90)
print("GNN-ADAPTGRAPH SCALABILITY EXPERIMENT")
print("=" * 90)

run_dataset(
    "DBLP",
    load_dblp
)

run_dataset(
    "Cora",
    load_cora
)

run_dataset(
    "PubMed",
    load_pubmed
)

print("\n" + "=" * 90)
print("SCALABILITY EXPERIMENT COMPLETED")
print("=" * 90)