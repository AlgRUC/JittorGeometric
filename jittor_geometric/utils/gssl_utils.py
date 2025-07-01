import jittor as jt
import math
import numpy as np
import random
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def index_to_mask(index, size):
    mask = jt.zeros(size, dtype=jt.bool)
    mask[index] = 1
    return mask


def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    num_nodes = label.shape[0]
    index = [i for i in range(num_nodes)]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(label.numpy() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx, size=num_nodes)
    val_mask = index_to_mask(val_idx, size=num_nodes)
    test_mask = index_to_mask(test_idx, size=num_nodes)
    return train_mask, val_mask, test_mask


def aug(graph, x, feat_drop_rate, edge_mask_rate):
    """
    Data augmentation function: Randomly drop features and mask edges.
    """
    ng = graph.clone()
    n_node = graph.num_nodes

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)


    src = graph.edge_index[0]
    dst = graph.edge_index[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng.edge_index = jt.stack([nsrc, ndst], dim=0)

    edge_index, edge_weight = ng.edge_index, ng.edge_attr

    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, n_node,
        improved=False, add_self_loops=True)
    with jt.no_grad():
        ng.csc = cootocsc(edge_index, edge_weight, n_node)
        ng.csr = cootocsr(edge_index, edge_weight, n_node)

    return ng, feat


def drop_feature(x, drop_prob):
    """
    Randomly drop features.
    """
    drop_mask = jt.rand((x.shape[1],)) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    """
    Randomly mask edges.
    """
    E = graph.num_edges

    mask_rates = jt.ones(E) * mask_prob
    masks = jt.bernoulli(1 - mask_rates)
    mask_idx = jt.nonzero(masks).squeeze(1)
    return mask_idx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(edge_index, num_nodes, alpha=0.2, self_loop=True):
    """
    Compute Personalized PageRank (PPR) using adjacency matrix.
    Args:
        edge_index (np.ndarray): [2, num_edges] array of edges.
        num_nodes (int): Number of nodes in the graph.
        alpha (float): Teleport probability in PPR.
        self_loop (bool): Whether to add self-loops to the adjacency matrix.
    Returns:
        np.ndarray: PPR matrix.
    """
    edge_index = edge_index.numpy()
    # Build sparse adjacency matrix from edge_index
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # Symmetrize
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])  # Add self-loops

    # Normalize adjacency matrix
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    # Compute PPR matrix
    ppr = alpha * inv((np.eye(norm_adj.shape[0]) - (1 - alpha) * norm_adj))
    return ppr
