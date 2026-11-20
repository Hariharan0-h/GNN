import networkx as nx


class GraphLoader:
    """
    Loads graph datasets for GNN-AdaptGraph.
    """

    @staticmethod
    def load_demo_graph():
        """
        Load Zachary's Karate Club graph.
        """
        return nx.karate_club_graph()

    @staticmethod
    def graph_statistics(G):
        print("=" * 50)
        print("Graph Statistics")
        print("=" * 50)

        print(f"Nodes                : {G.number_of_nodes()}")
        print(f"Edges                : {G.number_of_edges()}")

        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

        print(f"Average Degree       : {avg_degree:.2f}")

        print(f"Connected Components : {nx.number_connected_components(G)}")

        print("=" * 50)