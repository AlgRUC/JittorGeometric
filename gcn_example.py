import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid # 利用Planetoid类可以处理三个数据集，分别为“Cora”、“CiteSeer”和“PubMed”
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv, ChebConv, SGConv, GCN2Conv
# add by lusz
import time

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
# dataset='PubMed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
# add by lusz
total_forward_time = 0.0
total_backward_time = 0.0
# print(type(data))
# print(dataset[0])  Data(edge_index=[2, 10858], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

# print(data.edge_index)
# jt.Var([[   0    0    0 ... 2707 2707 2707]
        # [ 633 1862 2582 ...  165 1473 2706]], dtype=int32) 第一行为source,第二行为dist
    
# print(data.x) 
# jt.Var([[0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         ...
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]
#         [0. 0. 0. ... 0. 0. 0.]], dtype=float32)

# print(data.y) jt.Var([3 4 4 ... 3 3 3], dtype=int32)

# print(data.edge_attr)

# print(data.train_mask)
# print(data.val_mask)
# print(data.test_mask)
# jt.Var([ True  True  True ... False False False], dtype=bool)
# jt.Var([False False False ... False False False], dtype=bool)
# jt.Var([False False False ...  True  True  True], dtype=bool)

# GDC（Graph Diffusion Convolution，图扩散卷积）是一种图数据的预处理方法，旨在改进图卷积网络（GCN）的性能。
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

    def execute(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # print(edge_weight)
        x = nn.relu(self.conv1(x, edge_index, edge_weight)) # 传入feature,图拓扑
        x = nn.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return nn.log_softmax(x, dim=1)


model, data = Net(), data
optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    global total_forward_time, total_backward_time
    model.train()
    pred = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    jt.sync_all(True)
    # backward

    optimizer.step(loss)


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        y_ = data.y[mask]
        tmp = []
        for i in range(mask.shape[0]):
            if mask[i] == True:
                tmp.append(logits[i])
        logits_ = jt.stack(tmp)
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

jt.sync_all(True)
end = time.time()
# print(end - start)
print("epoch_time"+str(end-start))
print("total_forward_time"+str(total_forward_time))
print("total_backward_time"+str(total_backward_time))

