import math
import jittor as jt
from jittor import nn, Function
import numpy as np
from mpi4py import MPI
import threading
from pathlib import Path
import sys, os
import os.path as osp
import argparse
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
import time

sys.path.append("".join([str(Path(__file__).resolve().parent.parent.parent), "/jittor_geometric"]))
jt.flags.use_cuda = 1


class DistributedManager:
    _instance = None

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.world_size = self.size
        self.half_enabled = False
        self.csr_enabled = True
        self.time_all_record = {'to_np': 0, 'to_var': 0}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = DistributedManager()
        return cls._instance

    def broadcast_args(self, args):
        return self.comm.bcast(args, root=0)

    def all_reduce_sum(self, tensor):
        tensor_np = tensor.numpy()
        self.comm.Allreduce(MPI.IN_PLACE, tensor_np, op=MPI.SUM)
        tensor.update(jt.array(tensor_np))

    def all_to_all(self, output_list, input_list):
        send_counts = [t.numel() for t in input_list]

        all_send_counts = self.comm.allgather(send_counts)

        recv_counts = [all_send_counts[i][self.rank] for i in range(self.world_size)]

        send_displs = np.zeros(len(send_counts), dtype=int)
        for i in range(1, len(send_counts)):
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1]

        recv_displs = np.zeros(len(recv_counts), dtype=int)
        for i in range(1, len(recv_counts)):
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1]

        # 创建发送缓冲区
        flat_input = jt.concat([t.reshape(-1) for t in input_list]).numpy()

        # 创建接收缓冲区
        flat_output = np.empty(sum(recv_counts), dtype=np.float32)

        self.comm.Alltoallv(
            [flat_input, send_counts, send_displs, MPI.FLOAT],
            [flat_output, recv_counts, recv_displs, MPI.FLOAT]
        )

        offset = 0
        for i, count in enumerate(recv_counts):
            if count > 0:
                output_list[i].update(jt.array(flat_output[offset:offset + count]).reshape(output_list[i].shape))
            else:
                output_list[i].update(jt.zeros_like(output_list[i]))
            offset += count

    def synchronize_model_parameters(self, model):
        for param in model.parameters():
            param_np = param.numpy()
            self.comm.Allreduce(MPI.IN_PLACE, param_np, op=MPI.SUM)
            param.update(jt.array(param_np / self.size))

    def synchronize_model_parameters_non_blocking(self, model):
        requests = []
        for param in model.parameters():
            param_np = param.numpy()
            request = self.comm.Iallreduce(MPI.IN_PLACE, param_np, op=MPI.SUM)
            requests.append((param, param_np, request))
        return requests

    def finalize_synchronization(self, requests):
        for param, param_np, request in requests:
            request.Wait()
            param.update(jt.array(param_np / self.size))

    def broadcast(self, data, src=0):
        data_np = np.array(data)
        result = self.comm.bcast(data_np, root=src)
        return jt.array(result)

    def finalize(self):
        MPI.Finalize()


env = DistributedManager.instance()


def even_all_gather(tensor, env):
    """处理不同worker张量尺寸不一致的情况"""
    world_size = env.world_size
    tensor_size = np.array(tensor.shape)

    # 收集所有worker的张量大小
    all_tensor_sizes = np.zeros((world_size, len(tensor_size)), dtype=int)
    all_tensor_sizes[env.rank] = tensor_size
    env.comm.Allgather(MPI.IN_PLACE, all_tensor_sizes)

    # 找到最大尺寸
    max_size = np.max(all_tensor_sizes, axis=0)

    # 填充张量
    pad_tensor = jt.zeros(tuple(max_size), dtype=tensor.dtype)
    slices = tuple(slice(0, s) for s in tensor_size)
    pad_tensor = pad_tensor.assign(tensor, slices)

    # 收集所有张量
    recv_list = [jt.zeros(tuple(max_size), dtype=tensor.dtype) for _ in range(world_size)]
    env.all_to_all(recv_list, [pad_tensor] * world_size)
    return recv_list


def split(local_feature):
    """切分特征并分发到所有worker"""
    world_size = env.world_size

    # 沿特征维度切分
    splits = jt.chunk(local_feature, world_size, dim=1)
    splits_contiguous = [split.contiguous() for split in splits]

    # 准备接收缓冲区
    recv_list = [jt.zeros_like(splits_contiguous[env.rank]) for _ in range(world_size)]

    # 执行all-to-all通信
    with jt.no_grad():
        env.all_to_all(recv_list, splits_contiguous)

    # 拼接结果 (保持论文中的[V*W, D/W]形状)
    recv_tensor = jt.concat(recv_list, dim=0)
    return recv_tensor


def gather(local_feature):
    """收集所有worker的特征并拼接"""
    world_size = env.world_size

    # 沿节点维度切分
    splits = jt.chunk(local_feature, world_size, dim=0)
    splits_contiguous = [split.contiguous() for split in splits]

    # 准备接收缓冲区
    recv_list = [jt.zeros_like(splits_contiguous[env.rank]) for _ in range(world_size)]

    # 执行all-to-all通信
    with jt.no_grad():
        env.all_to_all(recv_list, splits_contiguous)

    # 拼接结果 (恢复为[V, D]形状)
    recv_tensor = jt.concat(recv_list, dim=1)
    return recv_tensor


class DistNNLayer(Function):
    def execute(self, features, weight):
        self.features = features
        self.weight = weight
        z_local = jt.matmul(features, weight)
        return z_local

    def grad(self, grad_output):
        features = self.features
        weight = self.weight
        grad_features = jt.matmul(grad_output, weight.transpose(0, 1))
        grad_weight = jt.matmul(features.transpose(0, 1), grad_output)
        env.all_reduce_sum(grad_weight)
        return grad_features, grad_weight


class DistGraphLayer(Function):
    def execute(self, features, indptr, indices, data, layers, tag):
        self.save_vars = indptr, indices, data, layers, tag

        if tag == 0:
            features = split(features)

        output = spmm_csr(features, indptr, indices, data)

        if tag == layers - 1:
            output = gather(output)

        return output

    def grad(self, grad_output):
        indptr, indices, data, layers, tag = self.saved_vars

        if tag == layers - 1:
            grad_output = split(grad_output)

        grad_features = spmm_csr(grad_output, indptr, indices, data, transpose=True)

        if tag == 0:
            grad_features = gather(grad_features)

        return grad_features, None, None, None, None, None

from jittor_geometric.data import CSR
# 导入cuSPARSE操作
src = "".join([str(Path(__file__).resolve().parent.parent.parent), "jittor_geometric/ops", "cpp/spmmcsr_op.h"])
header = "".join([str(Path(__file__).resolve().parent.parent.parent), "jittor_geometric/ops", "cpp/spmmcsr_op.h"])
spmmcsr_op = jt.compile_custom_ops((src, header))


class SpmmCsrFunc(Function):
    def execute(self, x, csr, transpose):
        self.csr = csr
        self.transpose = transpose

        A_rows = csr.row_offset.shape[0] - 1
        A_cols = x.shape[0]

        feature_dim = x.shape[1]
        self.A_rows = A_rows
        self.A_cols = A_cols
        self.feature_dim = feature_dim

        output = jt.zeros((A_rows, feature_dim), dtype=x.dtype)

        spmmcsr_op.spmmcsr(
            output,
            x,
            csr.column_indices,
            csr.edge_weight,
            csr.row_offset,
            A_rows,
            A_cols
        ).fetch_sync()

        return output

    def grad(self, grad_output):
        grad_x = jt.zeros((self.A_cols, self.feature_dim), dtype=grad_output.dtype)

        spmmcsr_op.spmmcsr(
            grad_x,
            grad_output,
            self.csr.column_indices,
            self.csr.edge_weight,
            self.csr.row_offset,
            self.A_rows,
            self.A_cols
        ).fetch_sync()

        return grad_x, None


# def spmm_csr(matrix, indptr, indices, data, transpose=False):
#     csr = CSR(
#         row_offset=indptr,
#         column_indices=indices,
#         edge_weight=data
#     )
#     A_rows = indptr.shape[0] - 1

#     return SpmmCsrFunc.apply(matrix, csr, transpose)
def spmm_csr(matrix, indptr, indices, data, transpose=False):
    env = DistributedManager.instance()
    world_size = env.world_size
    V = indptr.shape[0] - 1
    V_dist = matrix.shape[0]

    if transpose:
        transposed_data = np.zeros_like(data)
        transposed_indptr = np.zeros(V + 1, dtype=np.int32)
        transposed_indices = np.zeros_like(indices)

        counts = np.zeros(V, dtype=np.int32)
        for i in indices:
            counts[i] += 1
        transposed_indptr[1:] = np.cumsum(counts)

        idx = 0
        for i in range(V):
            start, end = indptr[i], indptr[i + 1]
            for j in range(start, end):
                col = indices[j]
                pos = transposed_indptr[col] + counts[col]
                transposed_indices[pos] = i
                transposed_data[pos] = data[j]
                counts[col] -= 1

        indptr = transposed_indptr
        indices = transposed_indices
        data = transposed_data
        transpose = False

    F_local = matrix.shape[1]
    matrix_aggregated = matrix.reshape(world_size, V, F_local).transpose(1, 0, 2).reshape(V, -1)

    csr = CSR(
        row_offset=indptr,
        column_indices=indices,
        edge_weight=data
    )

    output_aggregated = SpmmCsrFunc.apply(matrix_aggregated, csr, transpose)

    output_distributed = output_aggregated.reshape(V, world_size, -1).transpose(1, 0, 2).reshape(V_dist, -1)

    return output_distributed


import numpy as np


class Graph:
    def __init__(self, features, edge_index, num_classes, labels, edge_weight=None):
        self.num_nodes = features.shape[0]

        self.features = jt.array(features)
        self.num_classes = num_classes
        self.labels = jt.array(labels)

        # 使用分桶排序的CSR构建
        data, indptr, indices = self.bucket_csr_adj(edge_index, edge_weight)
        self.adj_data = jt.array(data)
        self.adj_indptr = jt.array(indptr)
        self.adj_indices = jt.array(indices)

    def bucket_csr_adj(self, edge_index, edge_weight):
        n = self.num_nodes
        src = edge_index[0].astype(np.int32)
        dst = edge_index[1].astype(np.int32)
        num_edges = len(src)

        rows = np.empty(2 * num_edges + n, dtype=np.int32)
        cols = np.empty_like(rows)

        rows[:num_edges] = src
        rows[num_edges:2 * num_edges] = dst
        rows[2 * num_edges:] = np.arange(n)

        cols[:num_edges] = dst
        cols[num_edges:2 * num_edges] = src
        cols[2 * num_edges:] = np.arange(n)

        if edge_weight is None:
            data = np.ones(rows.shape[0], dtype=np.float32)
        else:
            data = np.empty(rows.shape[0], dtype=np.float32)
            data[:num_edges] = edge_weight
            data[num_edges:2 * num_edges] = edge_weight
            data[2 * num_edges:] = 1.0

        deg = np.bincount(rows, weights=data, minlength=n)
        deg = np.maximum(deg, 1.0)
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        norm_data = data * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

        # 用整数除法替代浮点除法
        bucket_size = min(5000, max(100, n // 100))
        num_buckets = (n + bucket_size - 1) // bucket_size

        bucket_ids = rows // bucket_size

        # 全局排序：先按桶ID排序，相同桶内按行排序
        # 使用NumPy复合排序 (桶ID + 行ID)
        sort_order = np.lexsort((rows, bucket_ids))

        sorted_rows = rows[sort_order]
        sorted_cols = cols[sort_order]
        sorted_data = norm_data[sort_order]

        row_counts = np.bincount(sorted_rows, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(row_counts)

        return sorted_data, indptr, sorted_cols


class TensplitGCN(nn.Module):
    def __init__(self, g, hidden_dim=16, nlayers=2):
        super().__init__()
        self.g = g
        in_dim, out_dim = g.features.size(1), g.num_classes

        self.nlayers = nlayers
        self.layers = nn.ParameterList()

        jt.set_global_seed(0)

        world_size = env.world_size
        in_dim_padded = (in_dim + world_size - 1) // world_size * world_size
        out_dim_padded = (out_dim + world_size - 1) // world_size * world_size

        self.layers.append(nn.Parameter(jt.rand(in_dim_padded, hidden_dim)))
        for i in range(1, nlayers - 1):
            self.layers.append(nn.Parameter(jt.rand(hidden_dim, hidden_dim)))
        self.layers.append(nn.Parameter(jt.rand(hidden_dim, out_dim_padded)))

        for weight in self.layers:
            nn.init.xavier_uniform_(weight)

    def execute(self, features):
        world_size = env.world_size

        # 特征维度填充
        padded_features = features
        if features.shape[1] % world_size != 0:
            pad_size = world_size - (features.shape[1] % world_size)
            padding = jt.zeros((features.shape[0], pad_size), dtype=features.dtype)
            padded_features = jt.concat([features, padding], dim=1)

        hidden_features = padded_features

        for i in range(self.nlayers):
            hidden_features = DistGraphLayer.apply(
                hidden_features,
                jt.array(self.g.adj_indptr),
                jt.array(self.g.adj_indices),
                jt.array(self.g.adj_data),
                self.nlayers,
                i
            )

        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = nn.relu(hidden_features)
                hidden_features = nn.dropout(hidden_features, p=0.5)

        # 移除输出填充
        if hidden_features.shape[1] > self.g.num_classes:
            hidden_features = hidden_features[:, :self.g.num_classes]

        return hidden_features


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Graph dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5)')
    parser.add_argument('--num_parts', type=int, default=4, help='Number of graph partitions')
    return parser.parse_args()


def accuracy(pred, label):
    pred = jt.argmax(pred, dim=1)
    correct = (pred[0] == label).sum().item()
    total = label.shape[0]
    return correct / total


def load_dataset(args):
    my_dir = osp.dirname(osp.realpath(__file__))
    path = osp.join(my_dir, '..', '..', 'data')

    if args.dataset in ['computers', 'photo']:
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, args.dataset, geom_gcn_preprocess=False)
    elif args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = OGBNodePropPredDataset(name=args.dataset, root=path)
    elif args.dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
        dataset = HeteroDataset(path, args.dataset)
    elif args.dataset in ['reddit']:
        dataset = Reddit(os.path.join(path, 'Reddit'))

    data = dataset[0]
    features = data.x.numpy()
    labels = data.y.numpy().astype(np.int32)
    edge_index = data.edge_index.numpy()
    edge_weight = data.edge_attr.numpy() if data.edge_attr is not None else None
    num_classes = dataset.num_classes

    return features, labels, edge_index, edge_weight, num_classes


if __name__ == "__main__":
    args = get_argparse()

    features, labels, edge_index, edge_weight, num_classes = load_dataset(args)

    g = Graph(features, edge_index, num_classes, labels, edge_weight)

    model = TensplitGCN(g, hidden_dim=args.hidden_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 前向
        output = model(g.features)

        loss = nn.cross_entropy_loss(output, g.labels)
        acc = accuracy(output, g.labels)

        # 步进
        optimizer.step(loss)

        # 参数同步
        requests = env.synchronize_model_parameters_non_blocking(model)
        env.finalize_synchronization(requests)

        epoch_time = time.time() - epoch_start

        if env.rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}:")
            print(f"  时间: {epoch_time:.4f}秒")
            print(f"  损失: {loss.item():.4f}, 准确率: {acc:.4f}")

    env.finalize()