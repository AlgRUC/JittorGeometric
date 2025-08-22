'''
Description:
Author: zhengyp
Date: 2025-07-13
'''

import os.path as osp
import argparse
import pdb

import jittor as jt
from jittor import nn
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)

from jittor_geometric.dataloader import RecsysDataLoader
from jittor_geometric.datasets import MovieLens1M, MovieLens100K, Yelp2018
from jittor_geometric.datasets.recsys import Hit, MRR, NDCG, Recall, DataStruct
from jittor_geometric.nn.models import LightGCN, SimGCL, XSimGCL, DirectAU
from tqdm import tqdm

# Setup configuration
jt.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="ml-1m", help='graph dataset')
parser.add_argument('--model', default="lightgcn", help='model name')
parser.add_argument('--embedding_size', type=int, default=64, help='size of embedding')
parser.add_argument('--nlayer', type=int, default=2, help='number of layers')
parser.add_argument('--num_epochs', type=int, default=300, help='Training epochs')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--eval_step', type=int, default=1, help='eval step')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight of regularization loss')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
args = parser.parse_args()
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')

# Load recommender system dataset
if args.dataset == 'ml-100k':
    dataset = MovieLens100K(root=path)
elif args.dataset == 'ml-1m':
    dataset = MovieLens1M(root=path)
elif args.dataset == 'yelp2018':
    dataset = Yelp2018(root=path)

data = dataset.get(0)
num_epochs = args.num_epochs + 1

# Create data loaders for training and evaluation
train_loader = RecsysDataLoader(
    edge_index = data.train_edge_index,
    num_items  = int(data.num_items),
    batch_size = 4096,
    num_neg    = 1,
    shuffle    = True,
)

if args.model.lower() == 'lightgcn':
    model = LightGCN(data.num_users, data.num_items, args.embedding_size, args.nlayer, data.train_edge_index, reg_weight=args.reg_weight)
elif args.model.lower() == 'simgcl':
    model = SimGCL(data.num_users, data.num_items, args.embedding_size, args.nlayer, data.train_edge_index, reg_weight=args.reg_weight)
elif args.model.lower() == 'xsimgcl':
    model = XSimGCL(data.num_users, data.num_items, args.embedding_size, args.nlayer, data.train_edge_index, reg_weight=args.reg_weight)
elif args.model.lower() == 'directau':
    model = DirectAU(data.num_users, data.num_items, args.embedding_size, data.train_edge_index)

optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(model, data, split="val", k=[10]):
    model.eval()
    with jt.no_grad():
        if split == "val":
            split_edge_index = data.val_edge_index
        elif split == "test":
            split_edge_index = data.test_edge_index
        elif split == "train":
            split_edge_index = data.train_edge_index
        else:
            raise ValueError(f"Unknown split: {split}")

        return evaluate_topk_from_edges(model, split_edge_index, data.train_edge_index, data.num_items, k=k)

def build_eval_result(scores, positive_u, positive_i, ks=[10]):
    topk_scores, topk_idx = jt.topk(scores, k=max(ks), dim=1, largest=True)
    pos_matrix = jt.zeros((scores.shape), dtype=jt.int)
    pos_matrix[positive_u, positive_i] = 1
    pos_len = pos_matrix.sum(dim=1, keepdims=True)
    pos_idx = pos_matrix.gather(1, topk_idx)
    result = jt.concat([pos_idx, pos_len], dim=1)
    return result

def evaluate_topk_from_edges(model, split_edge_index, train_edge_index, num_items, k=[10]):
    data_struct = DataStruct()
    test_users = jt.unique(split_edge_index[0]).numpy().tolist()

    pbar = tqdm(test_users, desc="Evaluating", unit="user")
    for uid in pbar:
        # Ground-truth items in evaluation set
        true_items = split_edge_index[1][split_edge_index[0] == uid].numpy().tolist()
        # Items already interacted in the training set
        known_items = train_edge_index[1][train_edge_index[0] == uid].numpy().tolist()

        num_items = int(num_items)
        user_tensor = jt.array([int(uid)], dtype="int32")
        scores = model.full_predict(user_tensor).reshape(-1, num_items)
        scores[:, known_items] = -1e9  # Filter out known items

        batch_result = build_eval_result(scores, [0] * len(true_items), true_items, ks=k)
        data_struct.update_tensor("rec.topk", batch_result.numpy())

    rec_mat = jt.Var(data_struct.get_tensor("rec.topk"))
    metrics = {
        **Hit(k=k).calculate_metric(rec_mat),
        **MRR(k=k).calculate_metric(rec_mat),
        **NDCG(k=k).calculate_metric(rec_mat),
        **Recall(k=k).calculate_metric(rec_mat),
    }
    return metrics

best_score = -float("inf")
patience_counter = 0
for epoch in range(1, num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")
    model.train()
    for users, pos_items, neg_items in train_loader:
        optimizer.zero_grad()
        loss = model(users, pos_items, neg_items)
        optimizer.step(loss)

    if epoch % args.eval_step == 0 or epoch == num_epochs:
        val_metrics = evaluate(model, data, split="val", k=[10])
        print(f"[Validation @ Epoch {epoch}]")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")

        current_score = val_metrics["ndcg@10"]
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best NDCG@10={best_score:.4f}")
                break

test_metrics = evaluate(model, data, split="test", k=[10])
print("Test result:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")
