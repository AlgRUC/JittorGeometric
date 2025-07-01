from typing import Any, Optional, Tuple, Union

import jittor
from jittor import Var

from jittor_geometric.typing import SparseVar


def is_jittor_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`jittor.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Var):
        if src.layout == jittor.sparse_coo:
            return True
        if src.layout == jittor.sparse_csr:
            return True
        if src.layout == jittor.sparse_csc:
            return True
    return False