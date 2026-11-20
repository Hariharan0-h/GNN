"""
Runtime Graph Controller

Loads the trained GraphSAGE model and
predicts runtime adaptation actions.
"""

import torch

from gnn.model import GraphSAGEModel


class RuntimeController:

    def __init__(self, checkpoint):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            GraphSAGEModel()
            .to(self.device)
        )

        self.model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=self.device
            )
        )

        self.model.eval()

        # Must match RuntimeDecisionEngine
        self.actions = {
            0: "NO_ACTION",
            1: "THREAD_ADJUST",
            2: "REBALANCE"
        }

    def predict(self, runtime_graph):

        runtime_graph = runtime_graph.to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                runtime_graph.x,
                runtime_graph.edge_index
            )

            prediction = torch.argmax(
                output,
                dim=1
            )

        return [
            self.actions[int(i)]
            for i in prediction
        ]