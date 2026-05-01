import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class VirtualNode(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()

        self.embedding = nn.Embedding(1, hidden_dim)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            for _ in range(num_layers - 1)
        ])

        self.dropout = nn.Dropout(dropout)

    def init_vn(self, batch):
        num_graphs = batch.max().item() + 1
        idx = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
        return self.embedding(idx)

    def add_to_nodes(self, x, vn_emb, batch):
        return x + vn_emb[batch]

    def update(self, x, vn_emb, batch, layer_idx):
        agg = global_add_pool(x, batch)
        new_vn = self.mlps[layer_idx](agg + vn_emb)
        return self.dropout(new_vn)