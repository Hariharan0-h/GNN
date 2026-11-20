"""
Enhanced GraphSAGE Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):

    def __init__(
        self,
        input_dim=8,
        hidden_dim=64,
        output_dim=3,
        dropout=0.30
    ):
        super().__init__()

        # First GraphSAGE layer
        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Second GraphSAGE layer
        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                32
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                output_dim
            )
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index
    ):

        # Layer 1
        h1 = self.conv1(
            x,
            edge_index
        )

        h1 = self.bn1(h1)

        h1 = F.relu(h1)

        h1 = F.dropout(
            h1,
            p=self.dropout,
            training=self.training
        )

        # Layer 2
        h2 = self.conv2(
            h1,
            edge_index
        )

        h2 = self.bn2(h2)

        # Residual connection
        h = F.relu(h2 + h1)

        # Classifier
        out = self.classifier(h)

        return out