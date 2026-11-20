from gnn.graph_loader import GraphLoader

graph = GraphLoader.load_dblp()

print("=" * 60)
print("DBLP Loaded")
print("=" * 60)

print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())