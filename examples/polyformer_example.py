import os
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
from jittor_geometric.nn import PolyFormerModel, get_data_load

class Net(nn.Module):
    def __init__(self, dataset, args):
        super(Net, self).__init__()
        self.net = PolyFormerModel(dataset, args)

    def execute(self):
        list_mat = data.list_mat
        x = self.net(list_mat)
        return x


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0005) 
parser.add_argument('--weight_decay', type=float, default=0.00005) 
parser.add_argument('--early_stopping', type=int, default=100) 
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--dprate', type=float, default=0.5)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--d_ffn', type=int, default=128)
parser.add_argument('--q', type=float, default=1.0)
parser.add_argument('--multi', type=float, default=1.0)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--nlayer', type=int, default=1)
parser.add_argument('--base', type=str, default='mono')
parser.add_argument('--dataset', type=str, default='minesweeper')
parser.add_argument('--use_cuda', type=int, default=1)

args = parser.parse_args()

args.idx_run = 0

jt.flags.use_cuda = args.use_cuda
print('use_cuda', jt.flags.use_cuda)

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

if args.dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, args.dataset)
else:
    raise ValueError(f"Dataset {args.dataset} is not supported.")
    
dataset, data = get_data_load(args, dataset)


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
print("PolyFormer on dataset {}: Best Val Acc: {:.4f}, Test Acc: {:.4f}".format(
    args.dataset, best_val_acc, test_acc))    







