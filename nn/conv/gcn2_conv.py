from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

from math import log

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

from ..inits import glorot


def addmm(input_, mat1, mat2, *, beta=1, alpha=1, out=None):
    return beta*input_ + alpha*(mat1 @ mat2)


class GCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` Var.
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None

        self.weight1 = jt.random((channels, channels))

        if shared_weights:
            self.weight2 = None
        else:
            self.weight2 = jt.random((channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None

    def execute(self, x: Var, x_0: Var, edge_index: Adj,
                edge_weight: OptVar = None) -> Var:
        """"""

        if self.normalize:
            if isinstance(edge_index, Var):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
        # propagate_type: (x: Var, edge_weight: OptVar)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        x.multiply(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]
        if self.weight2 is None:
            out = x.add(x_0)
            out = addmm(out, out, self.weight1, beta=1. - self.beta,
                        alpha=self.beta)
        else:
            out = addmm(x, x, self.weight1, beta=1. - self.beta,
                        alpha=self.beta)
            out += addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                         alpha=self.beta)
        return out

    def message(self, x_j: Var, edge_weight: Var) -> Var:
        # return edge_weight.view(-1, 1) * x_jd
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)
