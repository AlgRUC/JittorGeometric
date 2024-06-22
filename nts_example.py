import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid # 利用Planetoid类可以处理三个数据集，分别为“Cora”、“CiteSeer”和“PubMed”
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv, ChebConv, SGConv, GCN2Conv,GCNConvNts
# add by lusz
import time
from jittor import Var
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

# 生成edge_weight(待优化)
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

def coo_to_csr():
    return



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
        self.conv1 = GCNConvNts(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConvNts(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

    def execute(self):
        x, edge_index, edge_weight,csc = data.x, data.edge_index, data.edge_attr,data.csc
        # print(csc.column_offset)
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
    # backward
    start_backward = time.time()
    optimizer.step(loss)
    end_backward = time.time()
    backward_time = end_backward - start_backward
    total_backward_time += backward_time


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

