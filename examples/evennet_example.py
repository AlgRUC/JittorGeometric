'''
Author: ivam
Date: 2024-12-13 
Description: 
'''
import os.path as osp
import argparse

import jittor as jt
from jittor import nn
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
from jittor_geometric.nn import EvenNet
import time
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

jt.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--dataset', default="cora", help='graph dataset')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for PPR')
parser.add_argument('--K', type=int, default=10, help='number of coe')
parser.add_argument('--spmm', action='store_true', help='whether using spmm')
args = parser.parse_args()
dataset=args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess=False)
elif dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
    dataset = OGBNodePropPredDataset(name=dataset, root=path)
elif dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, dataset)
elif dataset in ['reddit']:
    dataset = Reddit(os.path.join(path, 'Reddit'))

data = dataset[0]
total_forward_time = 0.0
total_backward_time = 0.0
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=False)
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)


class Net(nn.Module):
    def __init__(self, dataset, dropout=0.5):
        super(Net, self).__init__()
        hidden = 64
        self.lin1 = nn.Linear(dataset.num_features, hidden)
        self.lin2 = nn.Linear(hidden, dataset.num_classes)
        
        self.prop = EvenNet(args.K, args.alpha)
        self.dropout = dropout

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        x = nn.dropout(x, self.dropout)
        x = nn.relu(self.lin1(x))
        x = nn.dropout(x, self.dropout)
        x = self.lin2(x)
        x = self.prop(x, csc, csr)
        
        return nn.log_softmax(x, dim=1)


model, data = Net(dataset), data
optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4) 

def train():
    global total_forward_time, total_backward_time
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
start = time.time()
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

jt.sync_all()
end = time.time()
print("Training_time"+str(end-start))