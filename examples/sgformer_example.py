import os.path as osp
import sys
import time
import argparse
import traceback
import random
import numpy as np
from sklearn.metrics import roc_auc_score

import jittor as jt
from jittor import nn
from jittor_geometric import transforms as T
from jittor_geometric.datasets import HeteroDataset
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.nn import SGFormerModel
    
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--dataset', type=str, default='tolokers')
parser.add_argument('--backbone', type=str, default='gcn')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--use_bn', type=bool, default=True)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--use_graph', type=bool, default=True)
parser.add_argument('--use_weight', type=bool, default=True)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--graph_weight', type=float, default=0.5)
parser.add_argument('--aggregate', type=str, default='add')
parser.add_argument('--use_act', action='store_true')
parser.add_argument('--idx_run', type=int, default=0)

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, dataset, args):
        super(Net, self).__init__()
        self.net = SGFormerModel(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=args.num_classes,
            tc_num_layers=args.num_layers,
            tc_num_heads=args.num_heads,
            tc_alpha=args.alpha,
            tc_dropout=args.dropout,
            tc_use_bn=args.use_bn,
            tc_use_residual=args.use_residual,
            tc_use_weight=args.use_weight,
            tc_use_act=args.use_act,
            use_graph=args.use_graph,
            graph_weight=args.graph_weight,
            aggregate=args.aggregate,
            gnn_module=args.backbone
        )

    def execute(self):
        x = self.net(data)
        return x

print('loading datset')
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
if args.dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, args.dataset)
else:
    raise ValueError(f"Dataset {args.dataset} is not supported.")

data = dataset[0]
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

args.num_classes = 1 if args.dataset.lower() in ["minesweeper", "tolokers", "questions"] else dataset.num_classes
args.num_features = dataset.num_features

if args.dataset.lower() in ["roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"]:
    if args.idx_run not in [0,1,2,3,4]:
        raise ValueError(f"idx_run must be in [0,1,2,3,4], but got {args.idx_run}")
    data.train_mask = dataset[0].train_mask[args.idx_run]
    data.val_mask = dataset[0].val_mask[args.idx_run]
    data.test_mask = dataset[0].test_mask[args.idx_run]
else:
    raise ValueError(f"Dataset {args.dataset} is not supported for fixed splits.")

model = Net(dataset, args)

optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    model.train()
    logits = model()[data.train_mask]
    label = data.y[data.train_mask]  
    if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
        loss = nn.binary_cross_entropy_with_logits(logits.squeeze(-1), label.to(jt.float))
    else:
        loss = nn.cross_entropy_loss(logits, label)
    optimizer.step(loss)

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
            pred = (logits[mask].squeeze(-1) > 0)
            acc = roc_auc_score(y_true=data.y[mask].cpu().numpy(), y_score=logits[mask].squeeze(-1).cpu().numpy())
            loss = nn.binary_cross_entropy_with_logits(logits[mask].squeeze(-1), data.y[mask])
            acc = accs.append(acc)
        else:
            y_ = data.y[mask] 
            logits_=logits[mask]
            pred, _ = jt.argmax(logits_, dim=1)
            acc = pred.equal(y_).sum().item() / mask.sum().item()
            accs.append(acc)
    
    return accs

train()
best_val_acc = test_acc = 0
start = time.time()
early_stopping_cnt = 0
for epoch in range(args.epochs):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        early_stopping_cnt = 0
    else:
        early_stopping_cnt += 1
    if early_stopping_cnt > args.early_stopping:
        print('Early stopping')
        break
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    if epoch % 1 == 0:
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
print('SGFormer on dataset {}, best val acc {:.4f}, test acc {:.4f}'.format(
    args.dataset, best_val_acc, test_acc))