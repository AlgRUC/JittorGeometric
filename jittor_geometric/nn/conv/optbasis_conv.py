'''
Description: 
Author: Yuhe Guo
Date: 2024-12-30
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

class OptBasisConv(Module):
    r"""Graph Neural Networks with Learnable and Optimal Polynomial Bases
    <https://openreview.net/pdf?id=UjQIoJv927>`_ paper.

    This class implements the OptBasisConv architecture, which implicitly utilize the optimal polynomial bases on each channel via three term recurrence propagation. 
    Check Algorithm 4 and Algorithm 5 in the paper for more details.

    Mathematical Formulation:
        Please refer to Algorithm 2, 4 and 5 in paper for more details.

    Args:
        K (int): Order of polynomial bases.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        n_channels (int): Number of signal channels to be filtered.
        **kwargs (optional): Additional arguments for the base `Module`.
    """
    def __init__(self, K: int, n_channels:int, spmm:bool=True,  **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OptBasisConv, self).__init__(**kwargs)
        
        self.K = K
        self.spmm = spmm
        self.n_channels = n_channels
        
        self.reset_parameters()

    def reset_parameters(self):
        t = jt.zeros(self.K+1)
        t[0] = 1
        t = t.repeat(self.n_channels, 1)
        self.alpha_params = jt.Var(t) 

    def three_term_prop(self, csr, last_h, second_last_h):
        rst = self.propagate_spmm(x=last_h, csr=csr)
        _t = jt.linalg.einsum('nh,nh->h',rst,last_h)
        rst = rst - jt.linalg.einsum('h,nh->nh', _t, last_h)
        _t = jt.linalg.einsum('nh,nh->h',rst,second_last_h)
        rst = rst - jt.linalg.einsum('h,nh->nh', _t, second_last_h)
        rst = rst / jt.clamp((jt.norm(rst,dim=0)),1e-8)
        return rst
    
    def execute(self, x, csr):
        blank_noise = jt.randn_like(x)*1e-5
        x = x + blank_noise
        h0 = x /  jt.clamp((jt.norm(x,dim=0)), 1e-8)
        rst = jt.zeros_like(h0)
        rst = rst + self.alpha_params[:,0] * h0

        last_h = h0
        second_last_h = jt.zeros_like(h0)

        for i in range(1, self.K+1):
            h_i = self.three_term_prop(csr, last_h, second_last_h)
            rst = rst + self.alpha_params[:,i] * h_i
            second_last_h = last_h
            last_h = h_i

        return rst

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
