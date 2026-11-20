from graph.partition_manager import PartitionManager
from runtime.acgr import ACGR


class Rebalancer:

    def __init__(self, num_partitions=4):

        self.num_partitions = num_partitions

        self.acgr = ACGR(
            num_partitions=num_partitions
        )

    def rebalance(
        self,
        graph,
        partitions=None
    ):

        if partitions is None:

            manager = PartitionManager(
                num_partitions=self.num_partitions
            )

            partitions = manager.partition(
                graph
            )

        return self.acgr.refine(
            graph,
            partitions
        )