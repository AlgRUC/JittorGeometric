import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T
from jittor_geometric.nn import SGConv
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

# Setup configuration
jt.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='graph dataset')
parser.add_argument('--spmm', action='store_true', help='whether using spmm')
args = parser.parse_args()
dataset=args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

# Load dataset
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())


# Prepare data and edge normalization
data = dataset[0]
total_forward_time = 0.0
total_backward_time = 0.0
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
# Convert to sparse matrix format
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

# SGC model with K-hop aggregation
class Net(nn.Module):
    def __init__(self, dataset, dropout=0.8):
        super(Net, self).__init__()
        self.conv1 = SGConv(in_channels=dataset.num_features, out_channels=64, K=2, spmm=args.spmm)
        self.conv2 = SGConv(in_channels=64, out_channels=dataset.num_classes, K=2, spmm=args.spmm)
        self.dropout = dropout

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return nn.log_softmax(x, dim=1)



# Initialize model and optimizer
model, data = Net(dataset), data
optimizer = nn.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


# Training function
def train():
    model.train()
    pred = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)


# Evaluation function
def test():
    model.eval()
    logits, accs = model(), []
    # Evaluate on train, val, test sets
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


# Training loop
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    # Track best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
