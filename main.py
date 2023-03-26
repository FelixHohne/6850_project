import torch 
import numpy as np 
import argparse
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
import torch.nn.functional as F
import os
from torch_sparse import spmm
import dataset
import models

"""
Note: may need to follow pythonpanda2's installation instructions for torch_sparse. 
ARM Macs are considered to be CPU installations. The instructions are for an older PyTorch version, so for 1.13.1, try this: 
i.e. for PyTorch 1.13.1:
MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-1.31.1+${cpu}.html

https://github.com/rusty1s/pytorch_scatter/issues/241

"""
# Guide to GraphSAINT sampling https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser("Large Scale Graph Learning Codes")
parser.add_argument('--dataset', type=str, default="Flickr")
parser.add_argument('--method', type=str)

args = parser.parse_args()
print(args)

path = f"{os.path.dirname(__file__)}/dataset/{args.dataset}"
dataset = dataset.load_dataset(args.dataset, path)

data = dataset[0]

print("Edge index")
print(data)
print(type(data.edge_index))
print(data.edge_index.shape)
row, col = data.edge_index


loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=dataset.processed_dir,
                                     num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.GNNNetwork(dataset.num_node_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs



for epoch in range(1, 51):
    loss = train()
    accs = test()
    # TODO: Double check which of the 3 types of accuracies EllipticBitcoin is missing
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train acc: {accs[0]:04f}, valid acc: {accs[1]:04f}')





