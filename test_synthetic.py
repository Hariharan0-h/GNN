from data.synthetic import SyntheticGraphGenerator

for i in range(5):

    G = SyntheticGraphGenerator.generate()

    print("=" * 50)
    print(f"Graph {i+1}")

    print("Nodes :", G.number_of_nodes())

    print("Edges :", G.number_of_edges())

    print("Connected Components :",
          len(list(__import__("networkx").connected_components(G))))