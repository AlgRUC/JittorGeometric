from typing import Optional
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor.nn import Linear
from jittor_geometric.nn.conv import MessagePassing

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

    This class implements the Simplified Graph Convolution (SGC) layer, which removes nonlinearities and collapses weight 
    matrices across layers to achieve a simplified and computationally efficient graph convolution.

    Mathematical Formulation:

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where:
        - :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency matrix with added self-loops.
        - :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` is its diagonal degree matrix.
        - :math:`K` controls the number of propagation steps.
        - The adjacency matrix can include other values than :obj:`1`, representing edge weights via the optional `edge_weight` variable.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        K (int, optional): Number of propagation steps. Default is `1`.
        bias (bool, optional): Whether to include a learnable bias term. Default is `True`.
        **kwargs (optional): Additional arguments for the `MessagePassing` class.
    """

    _cached_x: Optional[Var]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1, bias: bool = True, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.spmm=spmm
        self.reset_parameters()
    

    def reset_parameters(self):
        glorot(self.lin.parameters()[0])
        zeros(self.lin.parameters()[1])
        self._cached_adj_t = None
        self._cached_csc=None


    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        """Perform forward propagation."""
        
        for k in range(self.K):
            if self.spmm and jt.flags.use_cuda == 1:
                x = self.propagate_spmm(x=x, csr=csr)
            else:
                x = self.propagate_msg(x=x, csc=csc, csr=csr)
        return self.lin(x)

    # propagate by message passing
    def propagate_msg(self,x, csc: CSC, csr:CSR):
        out = aggregateWithWeight(x,csc,csr)  
        return out
    
    # propagate by spmm
    def propagate_spmm(self, x, csr:CSR):
        out = SpmmCsr(x,csr)  
        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
