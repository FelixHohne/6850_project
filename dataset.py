import torch 
import math 
import os
from ogb.nodeproppred import NodePropPredDataset
import networkx as nx
import pandas as pd  

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def export_as_edge_list(G):
    nx.write_edgelist(G, "G.edgelist", data=False)

def constructGraph(path_to_file, node_feat, num_nodes):
    df = pd.read_csv(path_to_file, sep=" ")
    edge_index = df.to_numpy().T
    graph = {'edge_index': edge_index, 'edge_feat': None, 
            'node_feat': node_feat, 'num_nodes': num_nodes}
    return graph 

class Dataset(object):
    def __init__(self, name):
        self.name = name 
        self.graph = {}
        self.label = None 
    
    def __len__(self):
        return 1 
            
    def __get__item(self, idx):
        return self.graph, self.label 
    
    def get_idx_split(self, tr_prop = 0.5, val_prop = 0.25):
        num_nodes = self.label.shape[0]
        indices = torch.randperm(num_nodes, device=device)
        
        highest_tr_node = math.floor(tr_prop * num_nodes) 
        highest_val_node = math.floor((tr_prop + val_prop) * num_nodes)

        return indices[0:highest_tr_node], indices[highest_tr_node:highest_val_node], indices[highest_val_node:]

def load_dataset(ogb_dataset_name):
    dataset = NodePropPredDataset(name = ogb_dataset_name)
    graph, label = dataset[0]

    nc_dataset = Dataset(ogb_dataset_name)
    nc_dataset.graph = graph 
    nc_dataset.label = label 

    node_prop_splits = dataset.get_idx_split()

    for key, value in nc_dataset.graph.items():
        print(key, value)
        print("_________")

    def generate_splits():
        return torch.tensor(node_prop_splits['train']), \
        torch.tensor(node_prop_splits['valid']), \
        torch.tensor(node_prop_splits['test']) 
    
    print("Edge index", type(nc_dataset.graph['edge_index']), nc_dataset.graph['edge_index'].shape)
    
    nc_dataset.get_idx_split = generate_splits
    return nc_dataset
 





