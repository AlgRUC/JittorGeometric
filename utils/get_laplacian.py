from typing import Optional

import jittor as jt
from jittor import Var
from jittor_geometric.utils import add_self_loops, remove_self_loops

from .num_nodes import maybe_num_nodes


def get_laplacian(edge_index, edge_weight: Optional[Var] = None,
                  normalization: Optional[str] = None,
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None):
    r""" Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (Var int32): The edge indices.
        edge_weight (Var, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (Var.dtype, optional): The desired data type of returned Var
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """

    if normalization is not None:
        assert normalization in ['sym', 'rw']  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = jt.ones((edge_index.size(1)), dtype=dtype)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    shape = list(edge_weight.shape)
    shape[0] = num_nodes
    deg = jt.zeros(shape)
    deg = jt.scatter(deg, 0, row, src=edge_weight, reduce='add')
    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = jt.concat([-edge_weight, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)

        for i in range(deg_inv_sqrt.shape[0]):
            if deg_inv_sqrt[i] == float('inf'):
                deg_inv_sqrt[i] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp

    return edge_index, edge_weight
