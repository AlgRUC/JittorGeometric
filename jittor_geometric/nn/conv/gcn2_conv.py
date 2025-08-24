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
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper.

    This class implements the GCNII layer, which combines initial residual connections and identity mapping 
    to enable deeper graph convolutional networks without oversmoothing. The layer supports both message-passing 
    and sparse matrix multiplication (SPMM) for efficient propagation.

    Mathematical Formulation:
    .. math::
        \mathbf{H}^{(l)} = (1 - \beta) \big( (1 - \alpha) \mathbf{H}^{(l-1)} + \alpha \mathbf{H}^{(0)} \big) +
        \beta \big( \mathbf{\Theta}_1 \mathbf{H}^{(l-1)} + \mathbf{\Theta}_2 \mathbf{H}^{(0)} \big)

    where:
        :math:`\mathbf{H}^{(l)}` is the node feature matrix at layer :math:`l`.
        :math:`\mathbf{H}^{(0)}` is the initial node feature matrix.
        :math:`\mathbf{\Theta}_1` and :math:`\mathbf{\Theta}_2` are learnable weight matrices.
        :math:`\alpha` controls the strength of the initial residual connection.
        :math:`\beta` balances feature aggregation and transformation.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        cached (bool, optional): If set to `True`, caches the normalized edge indices. Default is `False`.
        add_self_loops (bool, optional): If set to `True`, adds self-loops to the input graph. Default is `True`.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `False`.
        **kwargs (optional): Additional arguments for the base `MessagePassing` class.
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