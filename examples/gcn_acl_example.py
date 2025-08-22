# -*- coding: utf-8 -*-
import os.path as osp
import argparse
import jittor as jt
from jittor import nn
import sys,os
import os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
from jittor_geometric.nn.conv import GCNConv
import time
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
jt.flags.use_acl = 1

jt.misc.set_global_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='graph dataset')
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
jt.flags.use_acl = 0
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
jt.flags.use_acl = 1
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

class Net(nn.Module):
    def __init__(self, dataset, dropout=0.8):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256,spmm=False)
        self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes,spmm=False)
        self.dropout = dropout

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        
        x = nn.relu(self.conv1(x, csc, csr,A))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr,A)
        return nn.log_softmax(x, dim=0)

A = jt.zeros((v_num, v_num))
A[edge_index[0], edge_index[1]] = edge_weight

model, data = Net(dataset), data

optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4) 
def train():
    global total_forward_time, total_backward_time
    model.train()
    pred = model()
    label = data.y
    log_probs = nn.log_softmax(pred, dim=1)
    loss = -log_probs[jt.arange(pred.shape[0]), label].mean()
    optimizer.step(loss)


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        indices = jt.where(mask > 0)[0]
        y_ = data.y[indices] 
        logits_=logits[indices]
        pred, _ = jt.argmax(logits_, dim=1)
        acc = pred.equal(y_).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
total_epoch_time = 0  # 累加每个epoch时间
start = time.time()   # 训练总开始时间

best_val_acc = 0.0
test_acc = 0.
# warm_up = 3
# for i in range(warm_up):
#     train()
for epoch in range(1, 201):
    epoch_start = time.time()
    train()
    epoch_time = time.time() - epoch_start
    total_epoch_time += epoch_time  # 累加时间

    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Time: {:.4f}s'
    print(log.format(epoch, train_acc, best_val_acc, test_acc, epoch_time))

jt.sync_all()
end = time.time()

avg_time = total_epoch_time / 200
print("Average Epoch Time: {:.4f}s".format(avg_time))
