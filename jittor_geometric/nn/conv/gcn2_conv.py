from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

from math import log

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight

from ..inits import glorot


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

    def __init__(self, in_channels: int, out_channels: int, cached: bool = False, add_self_loops: bool = True, 
                 spmm: bool=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)

        self.cached = cached
        self._cached_edge_index = None

        self.weight1 = jt.random((in_channels, out_channels))
        self.weight2 = jt.random((in_channels, out_channels))

        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        

    def execute(self, x: Var, x_0: Var, csc: OptVar, csr: OptVar, alpha: float, beta: float) -> Var:        
        
        support = (1-beta) * (1-alpha) * x + beta * jt.matmul(x, self.weight1)
        initial = (1-beta) * (alpha) * x_0 + beta * jt.matmul(x_0, self.weight2)
        if self.spmm and jt.flags.use_cuda==1:
            out = self.propagate_spmm(x=support, csr=csr) + initial
        else:
            out = self.propagate_msg(x=support, csc=csc, csr=csr) + initial
        return out

    def propagate_msg(self, x, csc: CSC, csr: CSR):
        out = aggregateWithWeight(x, csc, csr)  
        return out
    
    def propagate_spmm(self, x, csr: CSR):
        out = SpmmCsr(x, csr)
        return out

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)