from typing import Optional
from jittor_geometric.typing import OptVar

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.utils import remove_self_loops, add_self_loops
from jittor_geometric.utils import get_laplacian

from ..inits import glorot, zeros


class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.
    """

    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = jt.ones((K, in_channels, out_channels))
        if bias:
            self.bias = jt.ones((out_channels,))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptVar, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptVar = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        # edge_weight.masked_fill((edge_weight == float('inf')).int32(), 0)
        # edge_weight.masked_fill((edge_weight == float('-inf')).int32(), 0)
        for i in range(edge_weight.shape[0]):
            if edge_weight[i] == float('inf'):
                edge_weight[i] = 0
        # print('edge_weight: ', edge_weight.shape,
        #       edge_weight.min(), edge_weight.max())
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def execute(self, x, edge_index, edge_weight: OptVar = None,
                batch: OptVar = None, lambda_max: OptVar = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `execute() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = Var([2.0])
        if not isinstance(lambda_max, Var):
            lambda_max = Var([lambda_max])
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        Tx_0 = x
        # Tx_1 = x  # Dummy.
        out = jt.matmul(Tx_0, self.weight[0])
        # print('self weight:', self.weight)
        if self.weight.size(0) > 1:
            # print('norm: ', norm.shape,
            #       norm.min(), norm.max())
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            # print('Tx_1: ', Tx_1.shape, Tx_1.min(), Tx_1.max())
            out = out + jt.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + jt.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.reshape(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)
