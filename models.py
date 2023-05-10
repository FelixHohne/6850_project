import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm


class GNNNetwork(nn.Module):
    def __init__(self, dataset_name, in_channels, hidden_channels,out_channels):
        super(GNNNetwork, self).__init__()
        self.dataset_name = dataset_name
        if dataset_name == "BarabasiAlbert":
            print("using barabasi-albert")
            self.linear_1 = torch.nn.Linear(in_channels, hidden_channels)
            self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
            self.conv1 = GATConv(hidden_channels, hidden_channels)
            self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
            self.linear_2 = torch.nn.Linear(hidden_channels, hidden_channels)
            self.batch_norm3 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.batch_norm4 = nn.BatchNorm1d(hidden_channels)
            self.linear_layer = torch.nn.Linear(4 * hidden_channels, out_channels)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.batch_norm = nn.BatchNorm1d(hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.linear_layer = torch.nn.Linear(2 * hidden_channels, out_channels)


    def forward(self, x0, edge_index):
        if self.dataset_name == "BarabasiAlbert":
            x = self.linear_1(x0)
            x = self.batch_norm1(x)
            intermediate_1 = x 
            x = self.conv1(x, edge_index)
            x = self.batch_norm2(x)
            intermediate_2 = x 
            x = self.linear_2(x)
            x = self.batch_norm3(x)
            intermediate_3 = x
            x = self.conv2(x, edge_index)
            x = self.batch_norm4(x)
            intermediate_4 = x 
            final_x = torch.cat([intermediate_1, intermediate_2, intermediate_3, intermediate_4], dim=-1)
            x = self.linear_layer(final_x)
            return x.log_softmax(dim=-1)
        else:

            x1 = self.conv1(x0, edge_index)
            x1 = self.batch_norm(x1)
            x1 = F.relu(x1)
            x2 = self.conv2(x1, edge_index)
            x3 = torch.cat([x1, x2], dim=-1)
            x4 = self.linear_layer(x3)
            return x4.log_softmax(dim=-1)

