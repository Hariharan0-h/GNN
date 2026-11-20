"""
GNN-AdaptGraph Execution Engine

Executes graph partitions according to controller actions.

Actions
-------
NO_ACTION
THREAD_ADJUST
REBALANCE
"""

import time


class ExecutionEngine:

    NO_ACTION = 0
    THREAD_ADJUST = 1
    REBALANCE = 2

    def __init__(self, num_partitions=4):
        self.num_partitions = num_partitions

    # ---------------------------------------------------------
    # Normalize partitions
    # ---------------------------------------------------------

    @staticmethod
    def normalize_partitions(partitions):

        normalized = []

        for partition in partitions:

            # Normal case: set/list of node IDs
            if isinstance(partition, set):
                normalized.append(set(partition))

            elif isinstance(partition, (list, tuple)):

                values = []

                for item in partition:

                    if isinstance(item, set):
                        values.extend(item)

                    elif isinstance(item, (list, tuple)):
                        values.extend(item)

                    else:
                        values.append(item)

                normalized.append(set(values))

            else:
                raise TypeError(
                    "Invalid partition type: "
                    f"{type(partition)}"
                )

        return normalized

    # ---------------------------------------------------------
    # Traverse partition
    # ---------------------------------------------------------

    @staticmethod
    def traverse_partition(
        graph,
        partition
    ):

        node_set = set(partition)

        processed_edges = 0

        for source in partition:

            for neighbor in graph.neighbors(source):

                if neighbor in node_set:
                    processed_edges += 1

        return processed_edges

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    def execute(
        self,
        graph,
        partitions,
        actions
    ):

        # -----------------------------------------------------
        # IMPORTANT:
        # Use the partitions supplied by the caller.
        #
        # This is essential for:
        #
        # GraphSAGE + ACGR
        #
        # because ACGR has already produced the refined
        # partitioning.
        # -----------------------------------------------------

        partitions = self.normalize_partitions(
            partitions
        )

        # -----------------------------------------------------
        # Validate partition count
        # -----------------------------------------------------

        if len(partitions) != self.num_partitions:

            raise ValueError(
                f"Expected {self.num_partitions} partitions, "
                f"received {len(partitions)}"
            )

        # -----------------------------------------------------
        # Validate actions
        # -----------------------------------------------------

        if len(actions) != len(partitions):

            raise ValueError(
                f"Number of actions ({len(actions)}) "
                f"does not match number of partitions "
                f"({len(partitions)})"
            )

        execution_actions = list(actions)

        # -----------------------------------------------------
        # Execute partitions
        # -----------------------------------------------------

        start = time.perf_counter()

        partition_results = []

        for partition, action in zip(
            partitions,
            execution_actions
        ):

            processed_edges = (
                self.traverse_partition(
                    graph,
                    partition
                )
            )

            partition_results.append({

                "action": action,

                "nodes": len(partition),

                "processed_edges":
                    processed_edges
            })

        elapsed = (
            time.perf_counter() - start
        )

        # -----------------------------------------------------
        # Total processed edges
        # -----------------------------------------------------

        total_edges = sum(
            result["processed_edges"]
            for result in partition_results
        )

        # -----------------------------------------------------
        # Throughput
        # -----------------------------------------------------

        throughput = (

            total_edges / elapsed

            if elapsed > 0

            else 0.0
        )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {

            "partitions":
                partitions,

            "actions":
                execution_actions,

            "partition_results":
                partition_results,

            "processed_edges":
                total_edges,

            "execution_time":
                elapsed,

            "throughput":
                throughput
        }