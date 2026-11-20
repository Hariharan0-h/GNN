"""
GNN-AdaptGraph Runtime Execution Engine

Executes graph traversal on individual partitions.

Actions
-------
0 = NO_ACTION
1 = THREAD_ADJUST
2 = REBALANCE

IMPORTANT
---------
This implementation intentionally avoids Python multiprocessing.

The current traversal implementation is Python/NetworkX based.
Therefore, thread/process counts are treated as runtime
configuration decisions rather than actual parallel execution.

This provides a fair controlled experiment between:

    Static baseline
        -> NO_ACTION

    GNN-AdaptGraph
        -> GNN-predicted action

The execution kernel remains identical so that the experiment
does not accidentally measure multiprocessing startup/serialization
overhead instead of graph traversal performance.
"""

import os
import time


class RuntimeExecutor:

    # --------------------------------------------------
    # Runtime action IDs
    # --------------------------------------------------

    NO_ACTION = 0
    THREAD_ADJUST = 1
    REBALANCE = 2

    # --------------------------------------------------
    # Constructor
    # --------------------------------------------------

    def __init__(self):

        cpu_count = os.cpu_count() or 1

        self.default_threads = max(
            1,
            min(4, cpu_count)
        )

    # ==================================================
    # Basic partition traversal
    # ==================================================

    @staticmethod
    def traverse_partition(
        graph,
        partition
    ):
        """
        Traverse one graph partition.

        Only edges whose source and destination both
        belong to the partition are counted.

        Returns
        -------
        int
            Number of processed internal edges.
        """

        node_set = set(partition)

        visited = set()

        processed_edges = 0

        # ----------------------------------------------
        # Traverse every connected component
        # ----------------------------------------------

        for source in partition:

            if source in visited:
                continue

            stack = [source]

            visited.add(source)

            while stack:

                node = stack.pop()

                for neighbor in graph.neighbors(node):

                    # ----------------------------------
                    # Count only internal partition edges
                    # ----------------------------------

                    if neighbor in node_set:

                        processed_edges += 1

                        if neighbor not in visited:

                            visited.add(neighbor)

                            stack.append(neighbor)

        return processed_edges

    # ==================================================
    # Execute one partition
    # ==================================================

    def execute_partition(
        self,
        graph,
        partition,
        action
    ):
        """
        Execute one partition according to the selected
        runtime action.

        The action controls the runtime configuration label,
        but the actual traversal kernel remains identical.

        This is intentional because Python/NetworkX traversal
        is not being used as a true parallel execution kernel.
        """

        # ----------------------------------------------
        # Action names
        # ----------------------------------------------

        action_names = {
            self.NO_ACTION: "NO_ACTION",
            self.THREAD_ADJUST: "THREAD_ADJUST",
            self.REBALANCE: "REBALANCE"
        }

        # ----------------------------------------------
        # Validate action
        # ----------------------------------------------

        if action not in action_names:

            raise ValueError(
                f"Unknown runtime action: {action}"
            )

        action_name = action_names[action]

        # ----------------------------------------------
        # Runtime configuration
        # ----------------------------------------------

        if action == self.NO_ACTION:

            threads = 1

        elif action == self.THREAD_ADJUST:

            threads = self.default_threads

        elif action == self.REBALANCE:

            threads = self.default_threads

        else:

            threads = 1

        # ----------------------------------------------
        # IMPORTANT:
        #
        # Do NOT create ProcessPoolExecutor here.
        #
        # The action is recorded as the runtime decision,
        # while the traversal kernel remains identical.
        # ----------------------------------------------

        start_time = time.perf_counter()

        processed_edges = self.traverse_partition(
            graph,
            partition
        )

        end_time = time.perf_counter()

        execution_time = (
            end_time - start_time
        )

        # ----------------------------------------------
        # Throughput
        # ----------------------------------------------

        if execution_time > 0:

            throughput = (
                processed_edges /
                execution_time
            )

        else:

            throughput = 0.0

        # ----------------------------------------------
        # Result
        # ----------------------------------------------

        return {

            "action": action_name,

            "action_id": action,

            "threads": threads,

            "processed_edges": processed_edges,

            "execution_time": execution_time,

            "throughput": throughput

        }