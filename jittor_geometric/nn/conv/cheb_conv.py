from typing import Optional
from jittor_geometric.typing import OptVar

import jittor as jt
from jittor import Var, nn
from jittor_geometric.nn.conv import MessagePassing
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

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            :obj:`\lambda_max` should be a :class:`jittor.Var` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`jittor_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*,
          maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, 
                 normalization: Optional[str] = 'sym', bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])

        if bias:
            self.bias = jt.zeros(out_channels)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            glorot(lin.weight)
        zeros(self.bias)

    def __norm__(self, edge_index: Var, num_nodes: Optional[int],
                 edge_weight: OptVar, normalization: Optional[str],
                 lambda_max: OptVar = None, dtype: Optional[int] = None,
                 batch: OptVar = None):

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Var):
            lambda_max = jt.array([lambda_max], dtype=dtype)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        inf_mask = edge_weight == float('inf')
        edge_weight = jt.where(inf_mask, jt.zeros_like(edge_weight), edge_weight)
        loop_mask = edge_index[0] == edge_index[1]
        edge_weight = jt.where(loop_mask, edge_weight - 1, edge_weight)

        return edge_index, edge_weight

    def execute(self, x: Var, edge_index: Var, edge_weight: OptVar = None,
                batch: OptVar = None, lambda_max: OptVar = None) -> Var:
        
        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Var, norm: Var)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def forward(self, x: Var, edge_index: Var, edge_weight: OptVar = None,
                batch: OptVar = None, lambda_max: OptVar = None) -> Var:
        """PyTorch Geometric style forward method alias for execute"""
        return self.execute(x, edge_index, edge_weight, batch, lambda_max)

    def message(self, x_j: Var, norm: Var) -> Var:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
