import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T
from jittor_geometric.nn import SGConv

jt.flags.use_cuda = 1

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SGConv(
            dataset.num_features, dataset.num_classes, K=2, cached=True)

    def execute(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return nn.log_softmax(x, dim=1)


model, data = Net(), data
# Only perform weight-decay on first convolution.
optimizer = nn.Adam(model.parameters(), lr=0.2, weight_decay=0.005)


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
        mask = mask
        tmp = []
        for i in range(mask.shape[0]):
            if mask[i] == True:
                tmp.append(logits[i])
        logits_ = jt.stack(tmp)
        pred, _ = jt.argmax(logits_, dim=1)
        acc = pred.equal(y_).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 101):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
