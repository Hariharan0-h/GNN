"""
Generic Graph Loader for GNN-AdaptGraph

Supports:
- Demo graph
- DBLP
- Twitter
"""

import os
import networkx as nx


class GraphLoader:

    @staticmethod
    def load_demo_graph():
        """
        Load Zachary's Karate Club graph.
        """
        return nx.karate_club_graph()

    @staticmethod
    def load_dblp():
        """
        Load DBLP author graph.

        The PyG DBLP dataset is converted into a
        NetworkX graph containing author-author
        relationships.
        """

        from torch_geometric.datasets import DBLP

        dataset = DBLP(
            root="data/DBLP"
        )

        data = dataset[0]

        # Author-paper relationships
        author_paper_edges = data[
            "author",
            "to",
            "paper"
        ].edge_index

        # Convert bipartite author-paper graph
        # into an author-author graph.
        G = nx.Graph()

        num_authors = data["author"].num_nodes

        G.add_nodes_from(
            range(num_authors)
        )

        # Build paper -> authors mapping
        paper_authors = {}

        src = author_paper_edges[0].tolist()
        dst = author_paper_edges[1].tolist()

        for author, paper in zip(src, dst):

            paper_authors.setdefault(
                paper,
                []
            ).append(author)

        # Connect authors who share a paper
        for authors in paper_authors.values():

            for i in range(len(authors)):

                for j in range(i + 1, len(authors)):

                    u = authors[i]
                    v = authors[j]

                    if u != v:
                        G.add_edge(u, v)

        print("=" * 60)
        print("DBLP Author Graph")
        print("=" * 60)

        print(
            "Nodes                :",
            G.number_of_nodes()
        )

        print(
            "Edges                :",
            G.number_of_edges()
        )

        return G

    @staticmethod
    def load_twitter():

        path = "data/twitter/twitter.edgelist"

        if os.path.exists(path):

            G = nx.read_edgelist(
                path,
                nodetype=int
            )

            return G

        else:

            print(
                "Twitter dataset not found."
            )

            print(
                "Using synthetic Twitter-like graph."
            )

            G = nx.barabasi_albert_graph(
                1000,
                5
            )

            return G

    @staticmethod
    def graph_statistics(G):

        print("=" * 50)
        print("Graph Statistics")
        print("=" * 50)

        print(
            f"Nodes                : "
            f"{G.number_of_nodes()}"
        )

        print(
            f"Edges                : "
            f"{G.number_of_edges()}"
        )

        if G.number_of_nodes() > 0:

            avg_degree = (
                sum(
                    dict(G.degree()).values()
                )
                /
                G.number_of_nodes()
            )

            print(
                f"Average Degree       : "
                f"{avg_degree:.2f}"
            )

        print(
            f"Connected Components : "
            f"{nx.number_connected_components(G)}"
        )

        print("=" * 50)