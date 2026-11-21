"""
Graph-level runtime controllers.

Input: the communication graph of the current partitioning. One node per partition
with 11 features (8 partition features + 3 global runtime features broadcast to every
node: work scale, communication fraction, window-time scale); an edge joins two
partitions that share cut edges.
Output: one action for the whole system: NO_ACTION, THREAD_ADJUST, REBALANCE (ACGR,
local repair) or REPARTITION (METIS, global repartitioning).

GraphSAGEController  2 x SAGEConv (mean aggregator) + BatchNorm + residual,
                     mean||max readout, MLP head
MLPController        identical, but the SAGEConv layers are replaced by per-partition
                     linear layers (no message passing): the ablation that isolates
                     the contribution of neighbourhood aggregation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

from runtime.environment import NUM_FEATURES, Observation

NUM_GLOBALS = 3
INPUT_DIM = NUM_FEATURES + NUM_GLOBALS
NUM_ACTIONS = 4


def observation_to_data(obs: Observation, label: int = None) -> Data:
    k = obs.x.shape[0]
    glob = np.broadcast_to(obs.globals.reshape(1, -1), (k, obs.globals.shape[0]))
    x = torch.from_numpy(np.ascontiguousarray(np.concatenate([obs.x, glob], axis=1), dtype=np.float32))
    data = Data(
        x=x,
        edge_index=torch.from_numpy(obs.edge_index).long(),
        edge_weight=torch.from_numpy(obs.edge_weight).float(),
    )
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)
    return data


class _Head(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, NUM_ACTIONS))

    def forward(self, h, batch):
        return self.net(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1))


class GraphSAGEController(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = dropout
        self.head = _Head(hidden, dropout)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.bn2(self.conv2(h1, edge_index))
        return self.head(F.relu(h2 + h1), batch)


class MLPController(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=64, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = dropout
        self.head = _Head(hidden, dropout)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.bn1(self.lin1(x)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.bn2(self.lin2(h1))
        return self.head(F.relu(h2 + h1), batch)


MODELS = {"GraphSAGE": GraphSAGEController, "MLP": MLPController}


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
