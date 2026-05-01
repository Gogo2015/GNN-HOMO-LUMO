import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.conv = GINEConv(mlp, train_eps=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        out = self.bn(out)
        out = self.dropout(out)
        return torch.relu(out + x)