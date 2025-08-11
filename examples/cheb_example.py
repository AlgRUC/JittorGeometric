import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T
from jittor_geometric.nn import ChebConv
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

jt.flags.use_cuda = 0

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)

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
        self.conv1 = ChebConv(data.num_features, 64, K=2)
        self.conv2 = ChebConv(64, dataset.num_classes, K=2)

    def execute(self, x, edge_index, edge_weight=None,
                batch=None, lambda_max=None):
        x = nn.relu(self.conv1(x, edge_index, edge_weight))
        x = nn.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return nn.log_softmax(x, dim=1)

    def message(self, x_j, norm):
        return norm.reshape(-1, 1) * x_j


model, data = Net(), data
optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=0),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    pred = model(data.x, edge_index, edge_weight)[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)


def test():
    model.eval()
    logits, accs = model(data.x, edge_index, edge_weight), []
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


# train()
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
