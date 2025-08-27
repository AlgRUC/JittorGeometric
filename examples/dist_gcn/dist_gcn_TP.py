import math
import jittor as jt
from jittor import nn, Function
import numpy as np
from mpi4py import MPI
import sys, os
import os.path as osp
import argparse
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
import time

jt.flags.use_cuda = 1


class DistributedManager:
    _instance = None

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.world_size = self.size

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = DistributedManager()
        return cls._instance

    def all_reduce_sum(self, tensor):
        tensor_np = tensor.numpy()
        recv_buffer = np.empty_like(tensor_np)
        self.comm.Allreduce(tensor_np, recv_buffer, op=MPI.SUM)
        return jt.array(recv_buffer)

    def finalize(self):
        MPI.Finalize()


env = DistributedManager.instance()


class AllReduceSum(jt.Function):
    def execute(self, x):
        return env.all_reduce_sum(x)

    def grad(self, grad_x):
        return grad_x


all_reduce_sum = AllReduceSum.apply

from jittor_geometric.data import CSR
from pathlib import Path

src = "/".join([str(Path(__file__).resolve().parent.parent.parent), "jittor_geometric/ops", "cpp/spmmcsr_op.cc"])
header = "/".join([str(Path(__file__).resolve().parent.parent.parent), "jittor_geometric/ops", "cpp/spmmcsr_op.h"])
spmmcsr_op = jt.compile_custom_ops((src, header))


class SpmmFunction(jt.Function):
    def execute(self, matrix, adj_indptr, adj_indices, adj_data, adj_t_indptr, adj_t_indices, adj_t_data):
        self.saved_tensors = (adj_t_indptr, adj_t_indices, adj_t_data)

        csr = CSR(row_offset=adj_indptr, column_indices=adj_indices, edge_weight=adj_data)
        A_rows, A_cols = adj_indptr.shape[0] - 1, matrix.shape[0]
        output = jt.zeros((A_rows, matrix.shape[1]), dtype=matrix.dtype)
        spmmcsr_op.spmmcsr(output, matrix, csr.column_indices, csr.edge_weight, csr.row_offset, A_rows, A_cols).fetch_sync()
        return output

    def grad(self, grad_output):
        adj_t_indptr, adj_t_indices, adj_t_data = self.saved_tensors

        csr_t = CSR(row_offset=adj_t_indptr, column_indices=adj_t_indices, edge_weight=adj_t_data)
        A_rows, A_cols = adj_t_indptr.shape[0] - 1, grad_output.shape[0]
        grad_matrix = jt.zeros((A_rows, grad_output.shape[1]), dtype=grad_output.dtype)
        spmmcsr_op.spmmcsr(grad_matrix, grad_output, csr_t.column_indices, csr_t.edge_weight, csr_t.row_offset, A_rows, A_cols).fetch_sync()

        return grad_matrix, None, None, None, None, None, None


spmm = SpmmFunction.apply

import jittor as jt
import numpy as np
import scipy.sparse as sp


class Graph:
    def __init__(self, features, edge_index, num_classes, labels, edge_weight=None):
        self.num_nodes = features.shape[0]
        self.features = jt.array(features)
        self.num_classes = num_classes
        self.labels = jt.array(labels)

        data, indptr, indices = self.bucket_csr_adj(edge_index, edge_weight)
        self.adj_data = jt.array(data)
        self.adj_indptr = jt.array(indptr)
        self.adj_indices = jt.array(indices)

        t_data, t_indptr, t_indices = self._transpose_csr(indptr, indices, data)
        self.adj_t_data = jt.array(t_data)
        self.adj_t_indptr = jt.array(t_indptr)
        self.adj_t_indices = jt.array(t_indices)

    def bucket_csr_adj(self, edge_index, edge_weight):
        n = self.num_nodes
        src = edge_index[0].astype(np.int32)
        dst = edge_index[1].astype(np.int32)
        num_edges = len(src)

        rows_coo = np.concatenate((src, dst))
        cols_coo = np.concatenate((dst, src))

        if edge_weight is None:
            data_coo = np.ones(2 * num_edges, dtype=np.float32)
        else:
            data_coo = np.concatenate((edge_weight, edge_weight)).astype(np.float32)

        self_loop_rows = np.arange(n, dtype=np.int32)
        self_loop_cols = np.arange(n, dtype=np.int32)
        self_loop_data = np.ones(n, dtype=np.float32)

        rows_final = np.concatenate((rows_coo, self_loop_rows))
        cols_final = np.concatenate((cols_coo, self_loop_cols))
        data_final = np.concatenate((data_coo, self_loop_data))

        adj = sp.coo_matrix((data_final, (rows_final, cols_final)), shape=(n, n), dtype=np.float32)

        deg = np.array(adj.sum(axis=1)).flatten()
        deg = np.maximum(deg, 1.0)
        deg_inv_sqrt = 1.0 / np.sqrt(deg)

        adj_csr = adj.tocsr()

        row_norm = deg_inv_sqrt[adj_csr.nonzero()[0]]
        col_norm = deg_inv_sqrt[adj_csr.nonzero()[1]]
        norm_data = adj_csr.data * row_norm * col_norm

        adj_csr.data = norm_data

        return adj_csr.data, adj_csr.indptr, adj_csr.indices

    def _transpose_csr(self, indptr, indices, data):
        V = indptr.shape[0] - 1

        counts = np.bincount(indices, minlength=V)

        transposed_indptr = np.zeros(V + 1, dtype=np.int32)
        transposed_indptr[1:] = np.cumsum(counts)

        transposed_indices = np.empty_like(indices)
        transposed_data = np.empty_like(data)

        current_pos = np.copy(transposed_indptr[:-1])
        for i in range(V):
            for j in range(indptr[i], indptr[i + 1]):
                col = indices[j]
                pos = current_pos[col]
                transposed_indices[pos] = i
                transposed_data[pos] = data[j]
                current_pos[col] += 1

        return transposed_data, transposed_indptr, transposed_indices


class TensorParallelGCNLayer(nn.Module):
    def __init__(self, g, in_features, out_features):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(jt.randn(in_features // env.world_size, out_features))
        nn.init.xavier_uniform_(self.weight)

    def execute(self, local_features):
        h_local = spmm(local_features,
                       self.g.adj_indptr, self.g.adj_indices, self.g.adj_data,
                       self.g.adj_t_indptr, self.g.adj_t_indices, self.g.adj_t_data)
        z_partial = jt.matmul(h_local, self.weight)
        return z_partial


class TensorParallelGCN(nn.Module):
    def __init__(self, g, hidden_dim=16, nlayers=2):
        super().__init__()
        self.g = g
        in_dim, out_dim = g.features.shape[1], g.num_classes
        self.padded_in_dim = math.ceil(in_dim / env.world_size) * env.world_size
        self.layer1 = TensorParallelGCNLayer(g, self.padded_in_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.layer2 = TensorParallelGCNLayer(g, hidden_dim, out_dim)

    def execute(self, features):
        padded_features = features
        if features.shape[1] < self.padded_in_dim:
            padding = jt.zeros((features.shape[0], self.padded_in_dim - features.shape[1]))
            padded_features = jt.concat([features, padding], dim=1)

        local_features_l1 = jt.chunk(padded_features, env.world_size, dim=1)[env.rank]
        z1_partial = self.layer1(local_features_l1)
        z1_synced = all_reduce_sum(z1_partial)
        a1 = nn.relu(z1_synced)
        a1 = self.dropout(a1)

        local_features_l2 = jt.chunk(a1, env.world_size, dim=1)[env.rank]
        z2_partial = self.layer2(local_features_l2)
        output = all_reduce_sum(z2_partial)
        return output


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Graph dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size')
    return parser.parse_args()


def accuracy(pred, label):
    if env.rank == 0:
        pred_labels = jt.argmax(pred, dim=1)[0]
        return (pred_labels == label).sum().item() / label.shape[0]
    return 0.0


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

    # data_bundle = None
    # if env.rank == 0: data_bundle = load_dataset(args)
    # data_bundle = env.comm.bcast(data_bundle, root=0)
    # features, labels, edge_index, edge_weight, num_classes = data_bundle
    features, labels, edge_index, edge_weight, num_classes = load_dataset(args)

    g = Graph(features, edge_index, num_classes, labels, edge_weight)
    model = TensorParallelGCN(g, hidden_dim=args.hidden_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    jt_features = jt.array(g.features)
    if args.dataset in ['ogbn-products']:
        my_labels = g.labels.squeeze(1)
    else:
        my_labels = g.labels
    for epoch in range(args.epochs):
        jt.sync_all(True)
        epoch_start = time.time()

        output = model(jt_features)
        loss = nn.cross_entropy_loss(output, g.labels)

        optimizer.step(loss)

        jt.sync_all(True)
        epoch_time = time.time() - epoch_start

        if env.rank == 0:
            acc = accuracy(output, g.labels)
            print(f"Epoch {epoch + 1}/{args.epochs}: "
                  f"Time: {epoch_time:.4f}s, "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {acc:.4f}")