import torch 
import math 
import os
from torch_geometric.datasets import Flickr, EllipticBitcoinDataset, Reddit
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset, download_url
from typing import Optional, Callable, List
import numpy as np 
import os.path as osp

device = torch.device("cpu")
 


def get_idx_split(label, tr_prop = 0.5, val_prop = 0.25):
    num_nodes = label.shape[0]
    indices = torch.randperm(num_nodes, device=device)

    highest_tr_node = math.floor(tr_prop * num_nodes) 
    highest_val_node = math.floor((tr_prop + val_prop) * num_nodes)

    train_indices = torch.zeros(label.size(), dtype=torch.bool)
    train_indices[indices[0:highest_tr_node]] = True 

    valid_indices = torch.zeros(label.size(), dtype=torch.bool)
    valid_indices[indices[highest_tr_node:highest_val_node]] = True 

    test_indices = torch.zeros(label.size(), dtype=torch.bool)
    test_indices[indices[highest_val_node:]] = True 

    return train_indices, valid_indices, test_indices

def erdos_renyi_as_graph_data(n, p, num_features, num_classes):
    G = nx.erdos_renyi_graph(n, p)
    num_nodes = n 
    X = torch.rand((num_nodes, num_features))
    label = torch.randint(0, num_classes, (num_nodes, ))
    edge_index = torch.tensor(nx.to_pandas_edgelist(G).to_numpy().T)
    train_idx, valid_idx, test_idx = get_idx_split(label)
    data = Data(x=X, edge_index=edge_index, y = label)
    data.train_mask = train_idx 
    data.valid_mask = valid_idx 
    data.test_mask = test_idx 
    data.num_node_features = num_nodes
    data.num_classes = num_classes
    return data 

def barabasi_albert(n, m):
    G = nx.barabasi_albert_graph(n, m)
    num_nodes = n 
    X = 1000 * torch.rand((num_nodes,1))
    node_features_dict = {}
    for node in G.nodes():
        node_features_dict[node] = X[node].item()
    nx.set_node_attributes(G, node_features_dict, 'feature')

    label = torch.zeros(num_nodes, dtype=int)

    label_values = torch.zeros(num_nodes)

    for node in G.nodes():
        self_factor = 0.5
        label_value = self_factor * G.nodes[node]['feature']
        assert torch.abs(label_value - self_factor * X[node]) < 1e-8
        neighbor_aggr = 0 
        for neighbor in G.neighbors(node):
            neighbor_aggr += (1 / (1 + G.degree[neighbor])) *  G.nodes[neighbor]['feature']
        label_value += 0.5 * neighbor_aggr
        label_values[node] = label_value 
    
    quantiles_to_get = torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8])
    q = torch.quantile(label_values, quantiles_to_get)
    # print("Median: ", torch.median(label_values))
    # print("Quantile median:", q[2])
    # print("Q:", q[0], q[1], q[2], q[3])
    
    for node in G.nodes():
        print(label_values[node])
        if label_values[node].item() <= q[0].item():
            # print("label value: ", label_values[node], q[0].item(), "label 0")
            label[node] = 0 
        elif label_values[node] <= q[1].item():
            label[node] = 1
            # print("label value: ", label_values[node], q[1].item(), "label 1")
        elif label_values[node] <= q[2].item():
            label[node] = 2
            # print("label value: ", label_values[node], q[2].item(), "label 2")
        elif label_values[node] <= q[3].item():
            label[node] = 3
            # print("label value: ", label_values[node], q[3].item(), "label 3")
        else:
            label[node] = 4
            # print("label value: ", label_values[node], "label 4")
    edge_index = torch.tensor(nx.to_pandas_edgelist(G).to_numpy().T)

    print(label)
    train_idx, valid_idx, test_idx = get_idx_split(label)
    data = Data(x=X, edge_index=edge_index, y = label)
    data.train_mask = train_idx 
    data.valid_mask = valid_idx 
    data.test_mask = test_idx 
    data.num_node_features = 1
    data.num_classes = 5
    return data 

class BarabasiAlbertDataset(InMemoryDataset):
    def __init__(self, root = 'dataset/BarabasiAlbert', transform=None, pre_transform=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''
        self.root = root
        super(BarabasiAlbertDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        files = ['barabasi_albert.pt']
        return files

    def process(self):
        data = barabasi_albert(2240, 3)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ErdosRenyiDataset(InMemoryDataset):
    def __init__(self, root = 'dataset/ErdosRenyi', transform=None, pre_transform=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''
        self.root = root
        super(ErdosRenyiDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        files = ['erdos_renyi.pt']
        return files

    def process(self):
        data = erdos_renyi_as_graph_data(1000, 0.5, 100, 10)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def load_dataset(dataset, path):
    if dataset == "Flickr":
        dataset = Flickr(path)
    elif dataset == "EllipticBitcoinDataset":
        dataset = EllipticBitcoinDataset(path)
    elif dataset == "ErdosRenyi":
        dataset = ErdosRenyiDataset()
    elif dataset == "BarabasiAlbert":
        dataset = BarabasiAlbertDataset(path)
    elif dataset == "Reddit":
        dataset = Reddit(path)
        print(dataset[0])
    else:
        raise ValueError
    return dataset 
