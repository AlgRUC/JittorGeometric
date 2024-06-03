from typing import Optional

import jittor as jt

from .num_nodes import maybe_num_nodes
from jittor import Var, init


def contains_self_loops(edge_index):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (Var int32): The edge indices.

    :rtype: bool
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr: Optional[Var] = None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (Var int32): The edge indices.
        edge_attr (Var, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`Var int32`, :class:`Var`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def segregate_self_loops(edge_index, edge_attr: Optional[Var] = None):
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (Var int32): The edge indices.
        edge_attr (Var, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`Var int32`, :class:`Var`, :class:`Var int32`,
        :class:`Var`)
    """

    mask = edge_index[0] != edge_index[1]
    inv_mask = jt.logical_not(mask)

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


def add_self_loops(edge_index, edge_weight: Optional[Var] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (Var int32): The edge indices.
        edge_weight (Var, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`Var int32`, :class:`Var`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = jt.arange(0, N, dtype=Var.int32)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = init.constant((N, ), edge_weight.dtype, fill_value)
        edge_weight = jt.concat([edge_weight, loop_weight], dim=0)

    edge_index = jt.concat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def add_remaining_self_loops(edge_index,
                             edge_weight: Optional[Var] = None,
                             fill_value: float = 1.,
                             num_nodes: Optional[int] = None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (Var int32): The edge indices.
        edge_weight (Var, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`Var int32`, :class:`Var`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = jt.arange(0, N, dtype=row.dtype)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = jt.concat([edge_index[:, mask], loop_index], dim=1)

    if edge_weight is not None:
        inv_mask = jt.logical_not(mask)
        loop_weight = init.constant((N, ), edge_weight.dtype, fill_value)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = jt.concat([edge_weight[mask], loop_weight], dim=0)

    return edge_index, edge_weight
