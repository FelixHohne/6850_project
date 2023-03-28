import networkx as nx 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import dataset
import scipy 
import os 
import copy
import os.path as osp
from typing import Optional
import torch
from tqdm import tqdm
from torch_sparse import SparseTensor

import dataset
import graph_sampler

use_metropolis_hastings = False 

G = nx.barabasi_albert_graph(224, 2)
path = f"{os.path.dirname(__file__)}/dataset/BarabasiAlbert"
barabasi_dataset = dataset.load_dataset("BarabasiAlbert", path)
data = barabasi_dataset[0]

if use_metropolis_hastings:
    loader = graph_sampler.MetropolisHastingsSampler(data, 47, 2)
else:
    loader = GraphSAINTRandomWalkSampler(data, batch_size=47, walk_length = 2)

sampled_results = loader._sample_nodes(40)

sampled_nodes = set()
for elem in sampled_results:
    sampled_nodes.add(elem.item())

print("Generated sampled nodes")
color_map = []
for node in G:
    if node in sampled_nodes:
        color_map.append('blue')
    else: 
        color_map.append('#8AC7DB')  

print("Preparing to run plotting algorithm")
plt.figure(3,figsize=(36, 36)) 
plt.title("Visualization of GraphSAINT Random Walk Sampler for Barabasi-Albert Graph")
pos = nx.kamada_kawai_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, node_color=color_map)
edges = nx.draw_networkx_edges(G, pos)

print("Saving figure")

plt.savefig(f'ab_224_2_is_metropolis_hastings_{use_metropolis_hastings}.png', bbox_inches='tight', dpi=300)


