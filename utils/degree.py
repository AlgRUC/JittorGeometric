from typing import Optional

import jittor as jt

from .num_nodes import maybe_num_nodes


def degree(index, num_nodes: Optional[int] = None,
           dtype: Optional[int] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    Var.
    """
    N = maybe_num_nodes(index, num_nodes)
    out = jt.zeros((N, ), dtype=dtype)
    one = jt.ones((index.size(0), ), dtype=out.dtype)
    return out.scatter_add_(0, index, one)
