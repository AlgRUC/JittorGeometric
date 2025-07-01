import jittor as jt
from jittor import Var
from jittor_geometric.utils import remove_self_loops, segregate_self_loops

from .num_nodes import maybe_num_nodes


def contains_isolated_nodes(edge_index, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (Var int32): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    (row, col), _ = remove_self_loops(edge_index)

    return jt.unique(jt.concat((row, col))).size(0) < num_nodes


def remove_isolated_nodes(edge_index, edge_attr=None, num_nodes=None):
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.

    Args:
        edge_index (Var int32): The edge indices.
        edge_attr (Var, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (Var int32, Var, Var bool)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = jt.zeros((num_nodes), dtype=Var.bool)
    mask[edge_index.view(-1)] = 1

    assoc = jt.full((num_nodes, ), -1, dtype=Var.int32)
    assoc[mask] = jt.arange(mask.sum())
    edge_index = assoc[edge_index]

    loop_mask = jt.zeros_like(mask)
    loop_mask[loop_edge_index[0]] = 1
    loop_mask = loop_mask & mask
    loop_assoc = jt.full_like(assoc, -1)
    loop_assoc[loop_edge_index[0]] = jt.arange(loop_edge_index.size(1))
    loop_idx = loop_assoc[loop_mask]
    loop_edge_index = assoc[loop_edge_index[:, loop_idx]]

    edge_index = jt.concat([edge_index, loop_edge_index], dim=1)

    if edge_attr is not None:
        loop_edge_attr = loop_edge_attr[loop_idx]
        edge_attr = jt.concat([edge_attr, loop_edge_attr], dim=0)

    return edge_index, edge_attr, mask
