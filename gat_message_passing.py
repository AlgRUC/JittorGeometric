'''
Author: lusz
Date: 2024-07-01 15:42:52
Description: 
'''
import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T
from jittor_geometric.nn import GATConv
import time
from jittor import Var
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from jittor_geometric.data import CSC,CSR
from jittor_geometric.ops import cootocsr,cootocsc

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, Var):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = jt.ones((edge_index.size(1), ))

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        shape = list(edge_weight.shape)
        shape[0] = num_nodes
        deg = jt.zeros(shape)
        deg = jt.scatter(deg, 0, col, src=edge_weight, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


jt.flags.use_cuda = 0

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--dataset', help='graph dataset')
args = parser.parse_args()
dataset=args.dataset

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

v_num = data.x.shape[0]
e_num = data.edge_index.shape[1]
edge_index, edge_weight=data.edge_index,data.edge_attr
jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        False, True)

with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features,64,e_num, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GATConv(64, dataset.num_classes,e_num, cached=True,
                             normalize=not args.use_gdc)

    def execute(self):
        x, csc =data.x , data.csc
        x = nn.relu(self.conv1(x, csc))
        x = nn.dropout(x)
        x = nn.relu(self.conv2(x,csc))
        return nn.log_softmax(x, dim=1)


model, data = Net(), data
optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=1e-4),
    dict(params=model.conv2.parameters(), weight_decay=1e-4)
], lr=5e-3)


def train():
    model.train()

    pred = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)

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



train()
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
