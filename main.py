import torch 
import argparse
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import torch.nn.functional as F
import os
import dataset
import models
import graph_sampler

device = torch.device('cpu')

parser = argparse.ArgumentParser("Large Scale Graph Learning Codes")
parser.add_argument('--dataset', type=str, default="ErdosRenyi")
parser.add_argument('--method', type=str)
parser.add_argument("--sampler", type=str, default="srw")

args = parser.parse_args()
print(args)

path = f"{os.path.dirname(__file__)}/dataset/{args.dataset}"
dataset = dataset.load_dataset(args.dataset, path)
data = dataset[0]

row, col = data.edge_index

if args.sampler == "srw":
    loader = GraphSAINTRandomWalkSampler(data, batch_size=100, walk_length = 1)
elif args.sampler == "mhrw":
    print("using mhrw")
    loader = graph_sampler.MetropolisHastingsRandomWalkSampler(data, batch_size = 100, budget = 2)

model = models.GNNNetwork(dataset.num_node_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
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


for epoch in range(1, 51):
    loss = train()
    accs = test()
    # # TODO: Double check which of the 3 types of accuracies EllipticBitcoin is missing
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train acc: {accs[0]:04f}, valid acc: {accs[1]:04f}')





