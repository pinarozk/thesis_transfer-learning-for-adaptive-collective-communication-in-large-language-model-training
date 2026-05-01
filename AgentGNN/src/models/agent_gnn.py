import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class StrongAgentGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=96, heads=4, dropout=0.15):
        super().__init__()

        self.dropout = dropout
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_in_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_in_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_in_dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)

        edge_repr_dim = hidden_dim * 2 + edge_in_dim

        self.shared_edge_mlp = nn.Sequential(
            nn.Linear(edge_repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.routing_head = nn.Linear(hidden_dim // 2, 1)
        self.scheduling_head = nn.Linear(hidden_dim // 2, 1)

    def encode(self, x, edge_index, edge_attr):
        h = self.node_proj(x)

        h1 = self.conv1(h, edge_index, edge_attr)
        h1 = self.norm1(h + h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.norm2(h1 + h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.norm3(h2 + h3)
        h3 = F.relu(h3)

        return h3

    def forward(self, x, edge_index, edge_attr):
        h = self.encode(x, edge_index, edge_attr)

        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_hidden = self.shared_edge_mlp(edge_repr)

        routing_logits = self.routing_head(edge_hidden).squeeze(-1)
        scheduling_pred = torch.sigmoid(self.scheduling_head(edge_hidden).squeeze(-1))

        return routing_logits, scheduling_pred