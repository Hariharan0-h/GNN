import torch
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv

print("PyTorch:", torch.__version__)
print("NetworkX:", nx.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("PyTorch Geometric: OK")

print("\nEnvironment setup successful!")