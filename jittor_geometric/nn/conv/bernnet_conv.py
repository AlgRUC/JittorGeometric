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
from scipy.special import comb
from ..inits import glorot, zeros, ones
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight

class BernNet(Module):
    r"""The graph propagation operator from the `"BernNet: Learning Arbitrary 
    Graph Spectral Filters via Bernstein Approximation"
    <https://arxiv.org/abs/2106.10994>`_ paper

    Mathematical Formulation:

    .. math::
        \mathbf{Z} = \sum_{k=0}^{K} \alpha_k \mathrm{Bern}_{k}(\tilde{L}) \mathbf{X}.

    where:
        :math:`\mathbf{X}` is the input node feature matrix.
        :math:`\mathbf{Z}` is the output node feature matrix.
        :math:`\mathrm{Bern}_{k}` is the Bernstein polynomial of order :math:`k`.
        :math:`\tilde{\mathbf{L}}` is the normalized Laplacian matrix of the graph, translated to the interval :math:`[-1,1]`.
        :math:`\alpha_k` is the parameter for the :math:`k`-th order Bernstein polynomial.

    Args:
        K (int): Order of polynomial, or maximum number of hops considered for message passing. 
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        **kwargs (optional): Additional arguments for the `MessagePassing` class.
    """


    def __init__(self, K: int, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BernNet, self).__init__(**kwargs)
        self.K = K
        self.spmm = spmm
        self.temp= jt.random((self.K + 1,))

        self.reset_parameters()

    def reset_parameters(self):
    	ones(self.temp)

    def execute(self, x: Var, csc1: OptVar, csr1: OptVar, csc2: OptVar, csr2: OptVar) -> Var:
        TEMP=nn.relu(self.temp)

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            if self.spmm and jt.flags.use_cuda==1:
                x = self.propagate_spmm(x=x, csr=csr2)
            else:
                x = self.propagate_msg(x=x, csc=csc2, csr=csr2)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            if self.spmm and jt.flags.use_cuda==1:
                x = self.propagate_spmm(x=x, csr=csr1)
            else:
                x = self.propagate_msg(x=x, csc=csc1, csr=csr1)
            for j in range(i):
                if self.spmm and jt.flags.use_cuda==1:
                    x = self.propagate_spmm(x=x, csr=csr1)
                else:
                    x = self.propagate_msg(x=x, csc=csc1, csr=csr1)
            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
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
