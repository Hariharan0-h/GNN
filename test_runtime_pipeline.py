from gnn.datasets.dblp import load_dblp

from controller.runtime_pipeline import RuntimePipeline

graph = load_dblp()

pipeline = RuntimePipeline()

runtime_graph = pipeline.process(graph)

print()

print("=" * 60)
print("Runtime Graph")
print("=" * 60)

print(runtime_graph)