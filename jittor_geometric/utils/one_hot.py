from typing import Optional

import jittor as jt
from jittor import Var


def one_hot(
    index: Var,
    num_classes: Optional[int] = None,
) -> Var:
    r"""Taskes a one-dimensional :obj:`index` var and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input var, in which case it will be :obj:`1`.

    Args:
        index (jittor.Var): The one-dimensional input var.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input var.
            (default: :obj:`None`)
        dtype (jittor.dtype, optional): The :obj:`dtype` of the output var.
    """
    if index.dim() != 1:
        raise ValueError("'index' var needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = jt.zeros((index.size(0), num_classes))
    return out.scatter_(1, index.unsqueeze(1), jt.Var([1]))