'''
Description: 
Author: ivam
Date: 2024-12-13
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar
import jittor as jt
import numpy as np
from jittor import Var,nn,Module
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight

class EvenNet(Module):
    r"""EvenNet: Ignoring Odd-Hop Neighbors Improves
    Robustness of Graph Neural Networks
    <https://arxiv.org/pdf/2205.13892>`_ paper.

    This class implements the EvenNet architecture, which improves the robustness of graph neural networks by focusing on even-hop neighbors while ignoring odd-hop neighbors. 

    Args:
        K (int): Maximum number of hops considered for message passing.
        alpha (float): Parameter controlling the weighting of different hops.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        **kwargs (optional): Additional arguments for the base `Module`.
    """

    #_cached_edge_index: Optional[Tuple[Var, Var]]
    #_cached_csc: Optional[CSC]
    def __init__(self, K: int, alpha: float, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(EvenNet, self).__init__(**kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K

        TEMP_jt = jt.array(TEMP)
        self.temp = nn.Parameter(jt.Var(TEMP_jt))

        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
        self.temp = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        self.temp[-1] = (1-self.alpha)**self.K
        

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        out = x * (self.temp[0])
        for k in range(self.K):
            if self.spmm and jt.flags.use_cuda==1:
                x = self.propagate_spmm(x=x, csr=csr)
            else:
                x = self.propagate_msg(x=x, csc=csc, csr=csr)
            if k // 2 == 1:
                out = out + self.temp[k+1] * x
        return out

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
