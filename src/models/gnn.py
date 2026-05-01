import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from .layers import GINELayer
from .virtual_node import VirtualNode


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in get_atom_feature_dims()
        ])

    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in get_bond_feature_dims()
        ])

    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


class GNN(nn.Module):
    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 300,
        dropout: float = 0.0,
        use_virtual_node: bool = False,
    ):
        super().__init__()

        self.use_virtual_node = use_virtual_node
        self.num_layers = num_layers

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        self.convs = nn.ModuleList([
            GINELayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

        if use_virtual_node:
            self.vn = VirtualNode(hidden_dim, num_layers, dropout)

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        vn_emb = self.vn.init_vn(batch) if self.use_virtual_node else None

        layer_outputs = []

        for i, conv in enumerate(self.convs):
            if self.use_virtual_node:
                x = self.vn.add_to_nodes(x, vn_emb, batch)

            x = conv(x, edge_index, edge_attr)
            layer_outputs.append(global_mean_pool(x, batch))

            if self.use_virtual_node and i < self.num_layers - 1:
                vn_emb = self.vn.update(x, vn_emb, batch, i)

        x = torch.stack(layer_outputs, dim=0).sum(dim=0)

        return self.mlp_out(x).squeeze(-1)