import torch 
import numpy as np 
import argparse
from torch_geometric.loader import GraphSAINTRandomWalkSampler, DataLoader, GraphSAINTNodeSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
import torch.nn.functional as F
import os
from torch_sparse import spmm
import dataset
import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("Large Scale Graph Learning Codes")
parser.add_argument('--dataset', type=str, default="ErdosRenyi")
parser.add_argument('--method', type=str)

args = parser.parse_args()
print(args)

built_in = ("Flickr", "EllipticBitcoinDataset")
if args.dataset in built_in:
    path = f"{os.path.dirname(__file__)}/dataset/{args.dataset}"
    dataset = dataset.load_dataset(args.dataset, path)
    data = dataset[0]
else:
    data = dataset.load_data(args.dataset)

row, col = data.edge_index

loader = GraphSAINTRandomWalkSampler(data, batch_size=25, walk_length = 1)

if args.dataset in built_in:
    model = models.GNNNetwork(dataset.num_node_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
else:
    model = models.GNNNetwork(data.num_node_features, hidden_channels=256, out_channels=data.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


counter = 0 

def train():
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        print(data) 
        # data = data.to(device)
        # optimizer.zero_grad()
        # out = model(data.x, data.edge_index)
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item() * data.num_nodes
        # total_examples += data.num_nodes
    # return total_loss / total_examples


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
    # accs = test()
    # # # TODO: Double check which of the 3 types of accuracies EllipticBitcoin is missing
    # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train acc: {accs[0]:04f}, valid acc: {accs[1]:04f}')





