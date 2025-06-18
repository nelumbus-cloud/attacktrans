# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class GCN(torch.nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN).
    - hidden_channels: number of hidden units in the first GCN layer
    - dropout: dropout rate (applied before/after each layer)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x




class GAT(torch.nn.Module):
    """
    Two-layer Graph Attention Network (GAT).
    - hidden_channels: output size per attention head in the first layer
    - heads: number of attention heads in the first layer
    - dropout: dropout rate (applied before/after each layer)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x
