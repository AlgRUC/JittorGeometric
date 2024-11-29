import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid # 利用Planetoid类可以处理三个数据集，分别为“Cora”、“CiteSeer”和“PubMed”
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv, ChebConv, SGConv, GCN2Conv
import time

from jittor_geometric.jitgeo_loader import RandomNodeLoader
from jittor_geometric.jitgeo_loader import NeighborLoader

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
# dataset='PubMed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
print(path)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]



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
        self.conv1 = GCNConv(dataset.num_features, 16, cached=False,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=False,
                             normalize=not args.use_gdc)

    def execute(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = nn.relu(self.conv1(x, edge_index, edge_weight)) # 传入feature,图拓扑
        x = nn.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return nn.log_softmax(x, dim=1)


model, data = Net(), data

source_node = jt.Var([1,3,5,7,9,11,13,15,17,19])
dataloader = NeighborLoader(dataset, source_node, [3, 5], batch_size = 3)
#dataloader = RandomNodeLoader(dataset, num_parts=2, fixed=False)

optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

for i in dataloader:
    print('**********')
    print(i.node_map)
    print(i.central_nodes)
    print(i.edge_index)
    print(i.x)
    print(i.y)
    

exit()

def train():
    global total_forward_time, total_backward_time
    for i in dataloader:
        model.train()
        pred = model(i)[i.train_mask]
        label = i.y[i.train_mask]
        loss = nn.nll_loss(pred, label)
        jt.sync_all(True)
        # backward

        optimizer.step(loss)


def test():
    model.eval()
    logits, accs = model(data), []
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
for epoch in range(1, 11):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

# jt.sync_all(True)
# end = time.time()
# # print(end - start)
# print("epoch_time"+str(end-start))
# print("total_forward_time"+str(total_forward_time))
# print("total_backward_time"+str(total_backward_time))

