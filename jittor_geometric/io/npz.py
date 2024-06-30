from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

from jittor import Var
import jittor as jt
from jittor_geometric.data import Data
from jittor_geometric.utils import remove_self_loops
from jittor_geometric.utils import to_undirected as to_undirected_fn


def read_npz(path: str, to_undirected: bool = True) -> Data:
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = np.array(x)
    x = jt.array(x).float32()
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    row = jt.array(adj.row).int32()
    col = jt.array(adj.col).int32()
    edge_index = jt.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)

    if to_undirected:
        edge_index, _ = to_undirected_fn(edge_index, num_nodes=x.shape[0])

    y = jt.array(f['labels']).int32()

    return Data(x=x, edge_index=edge_index, y=y)