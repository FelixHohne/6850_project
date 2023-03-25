import torch 
import math 
import os
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Flickr, EllipticBitcoinDataset
import networkx as nx
import pandas as pd  

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_dataset(dataset, path):
    if dataset == "Flickr":
        dataset = Flickr(path)
    elif dataset =="EllipticBitcoinDataset":
        dataset = EllipticBitcoinDataset(path)
    else:
        raise ValueError
    return dataset 