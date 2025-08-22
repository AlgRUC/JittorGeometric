'''
Description:
Author: zhengyp
Date: 2025-07-13
'''

import os.path as osp
import argparse

import jittor as jt
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.datasets import Planetoid, Amazon
from jittor_geometric.nn.models import NetworkEmbeddingModel
from sklearn.metrics import roc_auc_score

# Enable CUDA for GPU acceleration
jt.flags.use_cuda = 1

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cora", help='graph dataset')
parser.add_argument('--model', default="LINE", help='model name')
parser.add_argument('--order', default="first", help='order of LINE model')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--num_epochs', type=int, default=300, help='Training epochs')
parser.add_argument('--embedding_size', type=int, default=64, help='size of embedding')
parser.add_argument('--walk_length', type=int, default=10, help='walk length')
parser.add_argument('--walk_size', type=int, default=5, help='walk length')
parser.add_argument('--walks_per_node', type=int, default=2, help='walks per node')
args = parser.parse_args()
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

# Load dataset based on type
if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset)
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset)

# Create link prediction split (train/val/test)
data = dataset.make_link_split(val_ratio=0.1, test_ratio=0.10)

# Initialize network embedding model
num_nodes = int(data.num_nodes)

# Initialize network embedding model
model = NetworkEmbeddingModel(args.dataset, num_nodes, args.embedding_size, method=args.model, line_order=args.order)
model.set_graph(edge_index=data.train_pos_edge_index, num_nodes=num_nodes, bidirectional=True)
model.register_pos_edges(
    getattr(data, "train_pos_edge_index", None),
    getattr(data, "val_pos_edge_index", None),
    getattr(data, "test_pos_edge_index", None),
    undirected=True
)
model.prepare_walk_engine(walk_length=args.walk_length, walks_per_node=args.walks_per_node)
optimizer = jt.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training loop
for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0
    batch_count = 0
    
    # Train on batches of positive and negative edges
    for pos_i, pos_j, neg_j in model.batch_generator(batch_size=4096, window_size=args.walk_size, num_neg=1, shuffle=True):
        loss = model(pos_i, pos_j, neg_j)
        optimizer.step(loss)
        train_loss += float(loss.data[0])
        batch_count += 1

    # Validation phase
    model.eval()
    with jt.no_grad():
        val_pos = data.val_pos_edge_index
        val_neg = model.sample_neg_for_edges(val_pos, ratio=1, seed=epoch, undirected=True)

        pos_logits = model.edge_scores(val_pos)
        neg_logits = model.edge_scores(val_neg)

        y_true = jt.concat([jt.ones_like(pos_logits), jt.zeros_like(neg_logits)], dim=0).numpy()
        y_pred = jt.sigmoid(jt.concat([pos_logits, neg_logits], dim=0)).numpy()

        val_auc = roc_auc_score(y_true, y_pred)
        print(f'Epoch: {epoch}, train loss: {(train_loss / batch_count):.4f}, val_auc: {val_auc:.4f}')

# Test phase
model.eval()
with jt.no_grad():
    test_pos = data.test_pos_edge_index
    test_neg = model.sample_neg_for_edges(test_pos, ratio=1, seed=9999, undirected=True)

    pos_logits = model.edge_scores(test_pos)
    neg_logits = model.edge_scores(test_neg)

    y_true = jt.concat([
        jt.ones_like(pos_logits),
        jt.zeros_like(neg_logits)
    ], dim=0).numpy()

    y_pred = jt.sigmoid(jt.concat([
        pos_logits, neg_logits
    ], dim=0)).numpy()

    test_auc = roc_auc_score(y_true, y_pred)

print(f"Test AUC: {test_auc:.4f}")

