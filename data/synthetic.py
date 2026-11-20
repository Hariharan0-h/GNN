"""
Synthetic graph generator.

Creates different graph topologies used for
GraphSAGE training.
"""

import random
import networkx as nx


class SyntheticGraphGenerator:

    @staticmethod
    def generate():

        graph_type = random.choice(
            [
                "ER",
                "BA",
                "WS"
            ]
        )

        n = random.randint(100, 300)

        if graph_type == "ER":

            p = random.uniform(0.02, 0.08)

            G = nx.erdos_renyi_graph(
                n=n,
                p=p
            )

        elif graph_type == "BA":

            m = random.randint(2, 5)

            G = nx.barabasi_albert_graph(
                n=n,
                m=m
            )

        else:

            k = random.choice([4, 6, 8])

            p = random.uniform(
                0.10,
                0.40
            )

            G = nx.watts_strogatz_graph(
                n=n,
                k=k,
                p=p
            )

        return G