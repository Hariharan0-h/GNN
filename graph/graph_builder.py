"""
Graph Builder

Converts partition features into the
GraphSAGE input graph.
"""

from graph.communication_graph import CommunicationGraph


class GraphBuilder:

    def __init__(self):

        self.builder = CommunicationGraph()

    def build(self, features):

        graph = self.builder.build(features)

        return graph