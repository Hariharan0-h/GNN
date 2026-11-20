from controller.runtime_controller import RuntimeController
from graph.partition_manager import PartitionManager
from monitor.feature_extractor import FeatureExtractor
from graph.communication_graph import CommunicationGraph


class RuntimePipeline:

    def __init__(self, checkpoint):

        self.partition_manager = PartitionManager(
            num_partitions=4
        )

        self.feature_extractor = FeatureExtractor()

        self.controller = RuntimeController(
            checkpoint
        )

    def process(self, graph):

        # -----------------------------------------
        # 1. Partition Graph
        # -----------------------------------------

        print("Partitioning Graph...")

        partitions = self.partition_manager.partition(
            graph
        )

        print(
            f"Number of partitions: {len(partitions)}"
        )

        for i, partition in enumerate(partitions):

            print(
                f"Partition {i + 1}: "
                f"{len(partition)} nodes"
            )

        # -----------------------------------------
        # 2. Extract Runtime Features
        # -----------------------------------------

        print(
            "Extracting Runtime Features..."
        )

        features = (
            self.feature_extractor
            .extract_partition_features(
                graph,
                partitions
            )
        )

        print(
            "Feature matrix shape:",
            features.shape
        )

        # -----------------------------------------
        # 3. Build Communication Graph
        # -----------------------------------------

        print(
            "Building Communication Graph..."
        )

        communication_builder = CommunicationGraph()

        runtime_graph = (
            communication_builder.build(
                graph,
                partitions,
                features
            )
        )

        print(
            "Communication edge_index shape:",
            runtime_graph.edge_index.shape
        )

        print(
            "Communication edges:",
            runtime_graph.edge_index.shape[1]
        )

        # -----------------------------------------
        # 4. GraphSAGE Controller
        # -----------------------------------------

        print(
            "Running GraphSAGE Controller..."
        )

        actions = self.controller.predict(
            runtime_graph
        )

        return actions