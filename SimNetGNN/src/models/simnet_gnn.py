import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool


class SimNetGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        global_dim=6,
        hidden_dim=128,
        heads=4,
        dropout=0.10,
    ):
        super().__init__()

        self.dropout = dropout

        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, edge_in_dim),
            nn.LayerNorm(edge_in_dim),
            nn.ReLU(),
        )

        self.conv1 = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_in_dim,
            concat=True,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_in_dim,
            concat=True,
            dropout=dropout,
        )

        self.conv3 = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_in_dim,
            concat=True,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        readout_dim = hidden_dim * 3 + global_dim

        self.regressor = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, u):
        h = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        h1 = self.conv1(h, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2 + h1)   # residual
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.norm3(h3 + h2)   # residual
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        g_mean = global_mean_pool(h3, batch)
        g_max = global_max_pool(h3, batch)
        g_sum = global_add_pool(h3, batch)

        g = torch.cat([g_mean, g_max, g_sum, u], dim=-1)

        out = self.regressor(g).squeeze(-1)
        return out