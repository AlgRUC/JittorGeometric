import jittor as jt
from jittor import Var
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional

def coalesce(edge_index, 
             edge_weight: Optional[Var] = None, 
             num_nodes: Optional[int] = None, 
             reduce: str = 'sum', 
             is_sorted: bool = False, 
             sort_by_row: bool = True):

    num_edges = edge_index.shape[1]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = jt.zeros(num_edges + 1).astype(edge_index.dtype)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:] = idx[1:].mul(num_nodes).add(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = jt.sort(idx[1:], dim=0)
        edge_index = edge_index[:, perm]
        if edge_weight is not None:
            edge_weight = edge_weight[perm]

    mask = idx[1:] > idx[:-1]

    if jt.all(mask):
        return edge_index, edge_weight

    edge_index = edge_index[:, mask]

    if edge_weight is None:
        return edge_index, None
    else:
        num_edges = edge_index.shape[1]
        idx = jt.arange(0, num_edges, dtype=jt.int32)
        idx = idx - (~mask).cumsum(0)
        edge_weight = jt.zeros((num_edges,)).scatter_(0, idx, edge_weight, reduce=reduce)
        return edge_index, edge_weight