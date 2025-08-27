import os
import pickle
import os.path as osp
import argparse
import jittor as jt
from jittor import nn
import sys, os
import numpy as np
from pathlib import Path
from jittor_geometric.ops import DistSpmm
sys.path.append("".join([str(Path(__file__).resolve().parent.parent.parent), "/jittor_geometric"]))
from jittor_geometric.utils.distributed_utils import DistributedManager
from jittor_geometric.data import DistChunk

# enter the program
dist = DistributedManager()

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Graph dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate')
    parser.add_argument('--num_parts', type=int, default=4, help='Number of graph partitions')
    args = parser.parse_args()
    return args

args = get_argparse()

epochs = args.epochs
dataset = args.dataset

reorder_path = osp.join(str(Path(__file__).resolve().parent.parent.parent), 'data', 'reorder')
chunk_dir = osp.join(reorder_path, f'dist_{dataset}_{dist.size}part_chunks')
chunk_file = osp.join(chunk_dir, f'subgraph_{dist.size}_of_{dist.rank}')
# 检查数据分区目录是否存在
if not osp.exists(chunk_dir):
    if dist.rank == 0:
        print(f"\n[Error] Dataset directory not found: {chunk_dir}")
        print(f"Dataset '{dataset}' for {dist.size} parts does not exist, "
              f"please run 'python dist_partition.py --dataset {dataset} --num_parts {dist.size}' first.\n")
    
    dist.comm.Barrier()
    sys.exit(1)

with open(chunk_file, 'rb') as f:
    chunk_data = pickle.load(f)

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers = 2, spmm = False, dropout=0.8):
        super(Net, self).__init__()
        self.layers = layers
        self.weights = []

        if dist.rank == 0:
            self.weights.append(jt.random((in_dim, hidden_dim)))
            for _ in range(1, layers - 1):
                self.weights.append(jt.random((hidden_dim, hidden_dim)))
            self.weights.append(jt.random((hidden_dim, out_dim)))
        
        self.sync_parameters()
        self.dropout = dropout

    def sync_parameters(self):
        for i in range(self.layers):
            result = None
            if dist.rank == 0:
                result = self.weights[i]
            result = dist.broadcast(result, 0)
            if dist.rank != 0:
                self.weights.append(result)

    def execute(self, x, chunk_data):
        hidden_feature = x
        self.x = x
        self.hidden_features = []
        for i, weight in enumerate(self.weights):
            z = self.broadcast(chunk_data, hidden_feature)
            hidden_feature = z @ weight

            self.hidden_features.append(hidden_feature)


        return nn.log_softmax(hidden_feature, dim=1)

    def broadcast(self, chunk_data, local_data):
        result = DistSpmm(dist, local_data, chunk_data.chunk_CSR, local_vnum=int(chunk_data.v_num), trans_A=True)
        return result

local_feature = chunk_data.local_feature
model = Net(local_feature.shape[1], 16, chunk_data.num_classes, layers=2)
optimizer = nn.Adam(params=model.weights, lr=0.001, weight_decay=5e-4)

def acc(pred, local_label, train_mask, val_mask, test_mask):
    train_mask = jt.array(train_mask).stop_grad()
    val_mask = jt.array(val_mask).stop_grad()
    test_mask = jt.array(test_mask).stop_grad()

    pred_label = jt.argmax(pred, dim=1)
    
    train_correct = jt.array(pred_label[0][train_mask] == local_label[train_mask]).sum().item()
    train_total = train_mask.sum().item()

    val_correct = jt.array(pred_label[0][val_mask] == local_label[val_mask]).sum().item()
    val_total = val_mask.sum().item()

    test_correct = jt.array(pred_label[0][test_mask] == local_label[test_mask]).sum().item()
    test_total = test_mask.sum().item()

    metrics = [train_correct, train_total, val_correct, val_total, test_correct, test_total]
    metrics = np.array(metrics, dtype=np.float32)

    if dist.rank == 0:
        all_metrics = [metrics]
        for src in range(1, dist.size):
            recv_buf = np.zeros(6, dtype=np.float32)
            dist.comm.Recv(recv_buf, source=src)
            all_metrics.append(recv_buf)
        
        train_correct = 0
        train_sum = 0
        val_correct = 0
        val_sum = 0
        test_correct = 0
        test_sum = 0

        for i in range(0, dist.size):
            train_sum += all_metrics[i][1]
            train_correct += all_metrics[i][0]
            val_correct += all_metrics[i][2]
            val_sum += all_metrics[i][3]
            test_correct += all_metrics[i][4]
            test_sum += all_metrics[i][5]
        
        train_acc = train_correct / train_sum
        val_acc = val_correct / val_sum
        test_acc = test_correct / test_sum

        return [train_acc, val_acc, test_acc]
    
    else:
        dist.comm.Send(metrics, dest=0)

def train():
    model.train()
    train_mask = chunk_data.local_masks['train']
    local_label = chunk_data.local_label
    val_mask = chunk_data.local_masks['val']
    test_mask = chunk_data.local_masks['test']

    for epoch in range(0, epochs):
        epoch_start_time = time.time()

        pred = model(local_feature, chunk_data)

        train_loss = None

        if dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
            predictions = pred[jt.array(train_mask)]
            labels = chunk_data.local_label[jt.array(train_mask)]
            train_loss = nn.nll_loss(predictions, labels.reshape(-1))
        else: train_loss = nn.nll_loss(pred[jt.array(train_mask)], chunk_data.local_label[jt.array(train_mask)])

        optimizer.step(train_loss)
        jt.sync_all()
        
        epoch_end_time = time.time()

        acc_list = acc(pred, local_label, train_mask, val_mask, test_mask)
        print(f"{epoch:2d} | epoch time: {(epoch_end_time - epoch_start_time):.4f} | loss: {train_loss:.4f} | train acc: {(acc_list[0]):.4f} | val acc: {acc_list[1]:.4f} | test acc: {acc_list[2]:.4f}")


import time
dist.comm.Barrier()
if dist.rank == 0:
    print("\n======== Starting Training ========")
    start_time = time.time()

train()
dist.comm.Barrier()

if dist.rank == 0:
    end_time = time.time()
    duration = end_time - start_time
    print("======== Training Finished ========")
    print(f"Total training time: {duration:.2f} s on {dataset}_parted_to{dist.size}.")
