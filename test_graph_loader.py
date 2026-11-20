from data.load_graph import GraphLoader

# Load demo graph
graph = GraphLoader.load_demo_graph()

# Print graph statistics
GraphLoader.graph_statistics(graph)