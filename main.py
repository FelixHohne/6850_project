import torch
import argparse
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import torch.nn.functional as F
import os
import dataset
import models
import graph_sampler
import pandas as pd
import matplotlib.pyplot as plt


device = torch.device('cpu')

parser = argparse.ArgumentParser("Large Scale Graph Learning Codes")
parser.add_argument('--dataset', type=str, default="ErdosRenyi")
parser.add_argument("--sampler", type=str, default="srw")

args = parser.parse_args()
print(args)

path = f"{os.path.dirname(__file__)}/dataset/{args.dataset}"
dataset = dataset.load_dataset(args.dataset, path)
dataset.num_nodes = dataset.y.shape[0]
dataset.num_edges = dataset[0].edge_index.shape[1]
data = dataset[0]

row, col = data.edge_index


if dataset == "BarabasiAlbert":
    if args.sampler == "srw":
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=47, walk_length=2)
    elif args.sampler == "mhrw":
        print("executing mhrw")
        loader = graph_sampler.MetropolisHastingsRandomWalkSampler(
            data, batch_size=47, budget=2)
    elif args.sampler == "mhrwe":
        loader = graph_sampler.MetropolisHastingsRandomWalkWithEscapingSampler(
            data, batch_size=47, budget=2, alpha=0.25)
    elif args.sampler == "rcmh":
        loader = graph_sampler.RejectionControlMetropolisHastingsSampler(
            data, batch_size=47, budget=2, alpha=0.25)
else:
    if args.sampler == "srw":
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=100, walk_length=2)
    elif args.sampler == "mhrw":
        print("executing mhrw")
        loader = graph_sampler.MetropolisHastingsRandomWalkSampler(
            data, batch_size=100, budget=2)
    elif args.sampler == "mhrwe":
        loader = graph_sampler.MetropolisHastingsRandomWalkWithEscapingSampler(
            data, batch_size=100, budget=2, alpha=0.25)
    elif args.sampler == "rcmh":
        loader = graph_sampler.RejectionControlMetropolisHastingsSampler(
            data, batch_size=47, budget=2, alpha=0.25)

model = models.GNNNetwork(dataset.num_node_features, hidden_channels=256,
                          out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        print(data)
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


overall_accs = []
for epoch in range(1, 1000):
    loss = train()
    accs = test()
    epoch_dic = {
        "epoch" : epoch, 
        "loss": loss, 
        "train": accs[0], 
        "test": accs[-1]
    }
    overall_accs.append(epoch_dic)

    if args.dataset == "EllipticBitcoinDataset":
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train acc: {accs[0]:04f}, test acc: {accs[1]:04f}')
    else:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train acc: {accs[0]:04f}, valid acc: {accs[1]:04f}, test acc: {accs[2]:04f}' )


results_df = pd.DataFrame(
    overall_accs
)

results_df.to_csv(f"{args.dataset}_{args.sampler}.csv")

