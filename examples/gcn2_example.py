import os.path as osp

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCN2Conv
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from math import log
import argparse


jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--spmm', action='store_true', help='whether using spmm')
args = parser.parse_args()

dataset_name = 'cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

if dataset_name in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
else:
    # See more dataset examples in ./dataset_example.py
    pass

data = dataset[0]
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)


class Net(nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=64, alpha=0.1, lamda=0.5, dropout=0.6):
        super(Net, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(dataset.num_features, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, dataset.num_classes))

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, hidden_channels, spmm=args.spmm))

        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        _hidden = []
        x = nn.relu(self.lins[0](x))
        _hidden.append(x)

        for i, conv in enumerate(self.convs):
            x = nn.dropout(x, self.dropout, is_train=self.training)
            alpha = self.alpha
            beta = log(self.lamda / (i + 1) + 1)
            x = conv(x, _hidden[0], csc, csr, alpha, beta)
            x = nn.relu(x)

        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.lins[1](x)

        return nn.log_softmax(x, dim=-1)


model = Net(dataset, hidden_channels=64, num_layers=64, alpha=0.1, lamda=0.5, dropout=0.6)
optimizer = nn.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=0.01)


print(model)


def train():
    model.train()
    out = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(out, label)
    optimizer.step(loss)
    return float(loss)


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        y_ = data.y[mask] 
        logits_=logits[mask]
        pred, _ = jt.argmax(logits_, dim=1)
        acc = pred.equal(y_).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}')
