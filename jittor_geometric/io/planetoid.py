import sys
import os.path as osp
from itertools import repeat
import numpy as np
import jittor as jt
from jittor import Var
# from torch_sparse import coalesce as coalesce_fn, SparseTensor
from jittor_geometric.data import Data
from jittor_geometric.io import read_txt_array
from jittor_geometric.utils import remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = jt.arange(y.size(0), dtype=Var.int32)
    val_index = jt.arange(y.size(0), y.size(0) + 500, dtype=Var.int32)
    sorted_test_index = test_index.argsort()[1]

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = jt.zeros((len_test_indices, tx.size(1)))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = jt.zeros((len_test_indices, ty.size(1)))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = jt.zeros((len(graph) - allx.size(0), x.size(1)))
        tx_ext[sorted_test_index - allx.size(0)] = tx

        ty_ext = jt.zeros((len(graph) - ally.size(0), y.size(1)))
        ty_ext[sorted_test_index - ally.size(0)] = ty

        tx, ty = tx_ext, ty_ext

        x = jt.concat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        row, col, value = SparseTensor.from_dense(x).coo()
        rows, cols, values = [row], [col], [value]

        mask1 = index_to_mask(test_index, size=len(graph))
        mask2 = index_to_mask(jt.arange(allx.size(0), len(graph)),
                              size=len(graph))
        mask = jt.logical_or(jt.logical_not(mask1), jt.logical_not(mask2))
        isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.size(0):]

        rows += [isolated_index]
        cols += [jt.arange(isolated_index.size(0)) + x.size(1)]
        values += [jt.ones((isolated_index.size(0)))]

        x = SparseTensor(row=jt.concat(rows), col=jt.concat(cols),
                         value=jt.concat(values))
    else:
        x = jt.concat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]
    y = jt.concat([ally, ty], dim=0).argmax(dim=1)[0]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))
    if name == 'test.index':
        return read_txt_array(path, dtype=Var.int32)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    if isinstance(out, np.matrix):
        out = np.asarray(out)
    out = jt.array(out)
    return out

# careful


def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=False):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = jt.stack([jt.array(row), jt.array(col)], dim=0)
    if coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce_fn(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = jt.zeros((size, ), dtype=Var.bool)
    mask[index] = 1
    return mask
