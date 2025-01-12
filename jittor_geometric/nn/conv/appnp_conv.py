'''
Description: 
Author: ivam
Date: 2024-12-13
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar
import jittor as jt
from jittor import Var,nn,Module
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight

class APPNP(Module):
    r"""The graph propagation operator from the `"Predict then Propagate: 
    Graph Neural Networks meet Personalized PageRank"
    <https://arxiv.org/abs/1810.05997>`_ paper
    """
    #_cached_edge_index: Optional[Tuple[Var, Var]]
    #_cached_csc: Optional[CSC]
    def __init__(self, K: int, alpha: float, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(APPNP, self).__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        #self._cached_edge_index = None
        #self._cached_adj_t = None

        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
        pass
        #glorot(self.weight)
        #zeros(self.bias)
        #self._cached_adj_t = None
        #self._cached_csc=None

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        h = x
        for k in range(self.K):
            if self.spmm and jt.flags.use_cuda==1:
                x = self.propagate_spmm(x=x, csr=csr)
            else:
                x = self.propagate_msg(x=x, csc=csc, csr=csr)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        return x

    # propagate by message passing
    def propagate_msg(self,x, csc: CSC, csr:CSR):
        out = aggregateWithWeight(x,csc,csr)  
        return out
    
    # propagate by spmm
    def propagate_spmm(self, x, csr:CSR):
        out = SpmmCsr(x,csr)  
        return out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
