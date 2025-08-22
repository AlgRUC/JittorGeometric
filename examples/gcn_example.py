'''
Update:
- Include AUC calculation
- Add split index to train and test
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
from jittor_geometric.nn import GCNConv
import time
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from sklearn.metrics import roc_auc_score

# Setup configuration
jt.flags.use_cuda = 1
jt.misc.set_global_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='graph dataset')
parser.add_argument('--spmm', action='store_true', help='whether using spmm')
args = parser.parse_args()
dataset=args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

# Load dataset
if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(path, dataset)
elif dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
    dataset = OGBNodePropPredDataset(name=dataset, root=path)
elif dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, dataset)
elif dataset in ['reddit']:
    dataset = Reddit(os.path.join(path, 'Reddit'))

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

# Calculate AUC score for binary or multi-class classification
def calculate_auc(y_true, y_pred):
    try:
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            return roc_auc_score(y_true.numpy(), nn.softmax(y_pred, dim=1).numpy(), multi_class='ovr')
        else:
            return roc_auc_score(y_true.numpy(), nn.sigmoid(y_pred).numpy())
    except ValueError:
        pred = (y_pred > 0).int() if len(y_pred.shape) == 1 else jt.argmax(y_pred, dim=1)
        return pred.equal(y_true).sum().item() / len(y_true)

# GCN model with two conv layers
class Net(nn.Module):
    def __init__(self, dataset, dropout=0.8):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256,spmm=args.spmm)
        self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes,spmm=args.spmm)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return x

# Initialize model and optimizer
model, data = Net(dataset), data
optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4) 

# Training function
def train(split_idx=0):
    global total_forward_time, total_backward_time
    model.train()
    
    # Handle multiple splits
    if len(data.train_mask.shape) > 1:
        train_mask = data.train_mask[split_idx]
    else:
        train_mask = data.train_mask
    
    pred = model()[train_mask]
    label = data.y[train_mask]
    
    # Choose loss function based on task
    if dataset in ['questions', 'minesweeper']:
        loss = nn.binary_cross_entropy_with_logits(pred, label.float())
    else:
        loss = nn.cross_entropy_loss(pred, label)
    
    optimizer.step(loss)

# Evaluation function
def test(split_idx=0):
    model.eval()
    logits = model()
    accs = []
    
    # Evaluate on train, val, test sets
    masks = [data.train_mask, data.val_mask, data.test_mask]
    for mask in masks:            
        current_mask = mask[split_idx] if len(mask.shape) > 1 else mask
        y_true = data.y[current_mask]
        logits_masked = logits[current_mask]
        
        # Calculate metric based on task
        if dataset in ['questions', 'minesweeper']:
            acc = calculate_auc(y_true, logits_masked)
        else:
            pred, _ = jt.argmax(logits_masked, dim=1)
            acc = pred.equal(y_true).sum().item() / current_mask.sum().item()
        accs.append(acc)
    
    return accs

# Check for multiple data splits
has_multiple_splits = len(data.train_mask.shape) > 1
n_splits = data.train_mask.shape[0] if has_multiple_splits else 1

best_val_accs = [0] * n_splits
test_accs = [0] * n_splits
final_test_acc = 0
final_val_acc = 0

start = time.time()

# Train on each split
for split_idx in range(n_splits):
    print(f"\nTraining on split {split_idx + 1}/{n_splits}")
    model.reset_parameters()
    model.load_parameters(model.state_dict())
    optimizer = nn.Adam(model.parameters(), lr=0.01, weight_decay=0)
    
    best_val_acc = test_acc = 0
    
    # Training loop
    for epoch in range(1, 201):
        train(split_idx)
        train_acc, val_acc, tmp_test_acc = test(split_idx)
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        
        # Log progress
        if epoch % 20 == 0:
            metric_name = "AUC" if dataset in ['questions', 'minesweeper'] else "Acc"
            log = f'Split {split_idx + 1}/{n_splits}, Epoch: {{:03d}}, Train {metric_name}: {{:.4f}}, Val {metric_name}: {{:.4f}}, Test {metric_name}: {{:.4f}}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
    
    best_val_accs[split_idx] = best_val_acc
    test_accs[split_idx] = test_acc

# Calculate average results
final_val_acc = sum(best_val_accs) / n_splits
final_test_acc = sum(test_accs) / n_splits

jt.sync_all()
end = time.time()

# Print final results
metric_name = "AUC" if dataset in ['questions', 'minesweeper'] else "Acc"
print(f"\nFinal Results across {n_splits} splits:")
print(f"Average Val {metric_name}: {final_val_acc:.4f}")
print(f"Average Test {metric_name}: {final_test_acc:.4f}")
print(f"Training time: {end-start:.2f}s")

# Print per-split results if applicable
if has_multiple_splits:
    print("\nResults for each split:")
    for split_idx in range(n_splits):
        print(f"Split {split_idx + 1}: Val {metric_name}: {best_val_accs[split_idx]:.4f}, Test {metric_name}: {test_accs[split_idx]:.4f}")