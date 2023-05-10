import torch 
import numpy as np 
import argparse
import random
from torch_geometric.data import GraphSAINTSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree, add_self_loops, is_undirected, to_undirected
import torch.nn.functional as F
import os
from torch_sparse import spmm
import dataset
import models

class MetropolisHastingsRandomWalkSampler(GraphSAINTSampler):
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
        data['edge_index'] = to_undirected(add_self_loops(data['edge_index'])[0])
        print("is undirected:", is_undirected(data['edge_index']))
        super(MetropolisHastingsRandomWalkSampler,
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
                # print(f'ratio: {d_i / d_u}')
                if r < (d_i/d_u):
                    start[i] = u_idx
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))
    
    
class MetropolisHastingsRandomWalkWithEscapingSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).

    Args:    
        budget (int): The number of actions tracking an edge is denoted as total budget, 
        which is denoted as 𝐵 here. Usually, 𝐵 ≥ |𝑉′| as RW-based algorithms are likely 
        to backtrack when exploring the original graph.
        
        alpha (float): In RWE, random walker chooses a mode with probability 𝛼 to jump, and 1 − 𝛼 to walk. 
        If the random walker chooses walk, then a neighbor of current node is chosen as the next step in the walk. 
        If the jump mode is chosen, a node is chosen with a probability distribution.
    """
    def __init__(self, data, batch_size: int, budget: int, alpha: float,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir = None, log: bool = True, **kwargs):
        self.budget = budget
        self.alpha = alpha
        data['edge_index'] = add_self_loops(data['edge_index'])[0]
        super(MetropolisHastingsRandomWalkWithEscapingSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.budget}_'
                f'{self.alpha}_{self.sample_coverage}.pt')

    def _sample_nodes(self, batch_size):
        start = np.random.randint(self.N, size=batch_size)
        node_idx = list(start)
        for _ in range(self.budget):
            for i, node in enumerate(start):
                if random.uniform(0,1) < self.alpha:
                    start[i] = random.randint(0,self.N-1)
                else:
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
    

class RejectionControlMetropolisHastingsSampler(GraphSAINTSampler):
    r"""Rejection Control Metropolis-Hastings based on GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).

    Args:
        budget (int): The number of actions tracking an edge is denoted as total budget, 
        which is denoted as 𝐵 here. Usually, 𝐵 ≥ |𝑉′| as RW-based algorithms are likely 
        to backtrack when exploring the original graph.

        alpha (float): A number in [0, 1]. This parameter controls whether the sampler behaves more like the simple random walk (alpha = 0) or the Metropolis-Hastings random walk (alpha = 1).
    """
    def __init__(self, data, batch_size: int, budget: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir = None, log: bool = True, alpha: float = 0.5, **kwargs):
        self.budget = budget
        self.alpha = alpha
        data['edge_index'] = add_self_loops(data['edge_index'])[0]
        super(RejectionControlMetropolisHastingsSampler,
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

                if r <= (d_i/d_u)**self.alpha:
                    start[i] = u_idx
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))
    

class SimpleRandomWalkWithStallingSampler(GraphSAINTSampler):
    r"""Simple random walk with 1/2 probability of stalling at each node, based on GraphSAINT random walk sampler class (see
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
        data['edge_index'] = add_self_loops(data['edge_index'])[0]
        super(SimpleRandomWalkWithStallingSampler,
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
                r = random.uniform(0,1)

                if r <= 0.5:
                    start[i] = u_idx
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))
    
class SimpleRandomWalkWithEscapingSampler(GraphSAINTSampler):
    r"""Simple random walk with 1/2 probability of stalling at each node, based on GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).

    Args:
        budget (int): The number of actions tracking an edge is denoted as total budget, 
        which is denoted as 𝐵 here. Usually, 𝐵 ≥ |𝑉′| as RW-based algorithms are likely 
        to backtrack when exploring the original graph.

        alpha (float): In RWE, random walker chooses a mode with probability 𝛼 to jump, and 1 − 𝛼 to walk. 
        If the random walker chooses walk, then a neighbor of current node is chosen as the next step in the walk. 
        If the jump mode is chosen, a node is chosen with a probability distribution.
    """
    def __init__(self, data, batch_size: int, budget: int, alpha: float,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir = None, log: bool = True, **kwargs):
        self.budget = budget
        self.alpha = alpha
        data['edge_index'] = add_self_loops(data['edge_index'])[0]
        super(SimpleRandomWalkWithEscapingSampler,
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
                if random.uniform(0,1) < self.alpha:
                    start[i] = random.randint(0,self.N-1)
                else:
                    neighbors = self.adj[node.item()]
                    d_i = neighbors.storage.col().shape[0]
                    neighbor_idx = random.randint(0, d_i-1)
                    u_idx = neighbors.storage.col()[neighbor_idx].item()
                    start[i] = u_idx
                    
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))
    
class SimpleRandomWalkSampler(GraphSAINTSampler):
    r"""Simple random walk, based on GraphSAINT random walk sampler class (see
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
        data['edge_index'] = add_self_loops(data['edge_index'])[0]
        super(SimpleRandomWalkSampler,
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
                start[i] = u_idx
                    
                node_idx.append(start[i]) 
                
        return torch.from_numpy(np.array(node_idx))