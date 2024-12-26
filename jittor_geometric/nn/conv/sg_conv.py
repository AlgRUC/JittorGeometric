from typing import Optional
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor.nn import Linear
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.nn.conv.gcn_conv import gcn_norm


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` Var.

    """

    _cached_x: Optional[Var]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

    def execute(self, x: Var, edge_index: Adj,
                edge_weight: OptVar = None) -> Var:
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Var):
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache
        return self.lin(x)

    def message(self, x_j: Var, edge_weight: Var) -> Var:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
