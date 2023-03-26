import torch 
import numpy as np 
import argparse
import random
from torch_geometric.data import GraphSAINTSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
import torch.nn.functional as F
import os
from torch_sparse import spmm
import dataset
import models

class MetropolisHastingsSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).

    Args:
        budget (int): The number of actions tracking an edge is denoted as total budget, 
        which is denoted as 𝐵 here. Usually, 𝐵 ≥ |𝑉′| as RW-based algorithms are likely 
        to backtrack when exploring the original graph.
    """
    def __init__(self, data, batch_size: int, budget: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir = None, log: bool = True, **kwargs):
        self.budget = budget
        super(MetropolisHastingsSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.budget}_'
                f'{self.sample_coverage}.pt')

    def _sample_nodes(self, batch_size):
        start = np.random.randint(self.N, size=batch_size)
        node_idx = list(start)
        for _ in range(self.budget):
            for i, node in enumerate(start):
                neighbors = self.adj[node.item()]
                d_i = neighbors.storage.col().shape[0]
                neighbor_idx = random.randint(0, d_i-1)
                u_idx = neighbors.storage.col()[neighbor_idx].item()
                d_u = self.adj[u_idx].storage.col().shape[0]
                r = random.uniform(0,1)

                if r < (d_u/d_i):
                    start[i] = u_idx
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))