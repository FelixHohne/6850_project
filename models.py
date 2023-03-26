import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm


class GNNNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super(GNNNetwork, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear_layer = torch.nn.Linear(2 * hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.liner_layer.reset_parameters()
        self.batch_norm.reset_parameters()


    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x1 = self.batch_norm(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x3 = torch.cat([x1, x2], dim=-1)
        x4 = self.linear_layer(x3)
        return x4.log_softmax(dim=-1)

