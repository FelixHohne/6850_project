import torch 
import math 
import os
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Flickr, EllipticBitcoinDataset
import networkx as nx
import pandas as pd  
from torch_geometric.data import Data, InMemoryDataset, download_url
import pickle
from typing import Optional, Callable, List
import numpy as np 
import os.path as osp
from torch_geometric.loader import DataLoader



os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("cpu")

def load_dataset(dataset, path):
    if dataset == "Flickr":
        dataset = Flickr(path)
    elif dataset == "EllipticBitcoinDataset":
        dataset = EllipticBitcoinDataset(path)
    else:
        raise ValueError
    return dataset 

def load_data(data_name):
    if data_name == 'star_graph':
        return star_graph_as_pyg_data(100, 10, 3)
    else:
        raise ValueError

def get_idx_split(label, tr_prop = 0.5, val_prop = 0.25):
    num_nodes = label.shape[0]
    indices = torch.randperm(num_nodes, device=device)

    highest_tr_node = math.floor(tr_prop * num_nodes) 
    highest_val_node = math.floor((tr_prop + val_prop) * num_nodes)

    return indices[0:highest_tr_node], indices[highest_tr_node:highest_val_node], indices[highest_val_node:]


def erdos_renyi_as_graph_data(n, p, num_features, num_classes):
    G = nx.erdos_renyi_graph(n, p)
    print(G.number_of_nodes())
    num_nodes = n 
    X = torch.rand((num_nodes, num_features))
    label = torch.randint(0, num_classes, (num_nodes, ))
    edge_index = torch.tensor(nx.to_pandas_edgelist(G).to_numpy().T)
    print("We are edge:", edge_index)
    train_idx, valid_idx, test_idx = get_idx_split(label)
    data = Data(x=X, edge_index=edge_index, y = label)
    data.train_mask = train_idx 
    data.valid_mask = valid_idx 
    data.test_mask = test_idx 
    data.num_node_features = num_nodes

    return data 


# Generates the star graph with n + 1 edges and n nodes
def star_graph_as_pyg_data(n, num_features, num_classes):
    G = nx.star_graph(n)
    num_nodes = n + 1 # n + 1 nodes for star graph with n tentacles
    X = torch.rand((num_nodes, num_features))
    label = torch.randint(0, num_classes, (num_nodes,))
    print(torch.tensor(nx.to_pandas_edgelist(G).to_numpy()).T)
    edge_index = torch.tensor(nx.to_pandas_edgelist(G).to_numpy().T)

    train_idx, valid_idx, test_idx = get_idx_split(label)
    data = Data(x=X, edge_index=edge_index, y = label)
    data.train_mask = train_idx 
    data.valid_mask = valid_idx 
    data.test_mask = test_idx 

    return data 

    # @property 
    # def processed_file_names(self):
    #     return 'star_dataset.pt'
    
    # def download(self):
    #     path = osp.join(self.raw_dir, 'star.pkl')
    #     graph = star_graph_as_pyg_data(100, 10, 3)
    #     pickle.save(graph, path)
    
    # def process(self):
    #     load_path = osp.join(self.raw_dir, 'star.pkl')
    #     data = pickle.load(load_path)




class AmazonProducts(InMemoryDataset):
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    adj_full_id = '17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa'
    feats_id = '10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj'
    class_map_id = '1LIl4kimLfftj4-7NmValuWyCQE8AaE7P'
    role_id = '1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url.format(self.adj_full_id), self.raw_dir)
        print("download path is: ", path) 
        os.rename(path, osp.join(self.raw_dir, 'adj_full.npz'))

        path = download_url(self.url.format(self.feats_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'feats.npy'))

        path = download_url(self.url.format(self.class_map_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'class_map.json'))

        path = download_url(self.url.format(self.role_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'role.json'))

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])



    
    # def download(self):
    #     print("running download")
    #     data = star_graph_as_pyg_data(100, 10, 3)
        
    #     write_file = open(os.path.join(self.raw_dir, 'star.pkl'), 'w+')
    #     print("Comptued data and raw_file_path")
    #     pickle.dump(data, write_file)    
    #     print("Dumped pickle")
    #     pass 
    # def process(self):
    #     print(self.raw_paths[0])
    #     self.data = pickle.load(self.raw_paths[0])
    #     torch.save(self.data, 
    #                os.path.join(self.processed_dir, 'star_dataset.pt'))
        
    

# AmazonProducts(f"{os.path.dirname(__file__)}/dataset/{AmazonProducts}")