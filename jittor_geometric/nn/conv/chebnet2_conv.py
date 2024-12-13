'''
Description: 
Author: ivam
Date: 2024-12-13
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar
import jittor as jt
import math
import numpy as np
from jittor import Var,nn,Module
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from scipy.special import comb
from ..inits import glorot, zeros, ones
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebNetII(Module):
    r"""The graph propagation operator from the `"Convolutional Neural Networks
     on Graphs with Chebyshev Approximation, Revisited"
    <https://arxiv.org/abs/2202.03580>`_ paper
    """

    #_cached_edge_index: Optional[Tuple[Var, Var]]
    #_cached_csc: Optional[CSC]
    def __init__(self, K: int, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebNetII, self).__init__(**kwargs)
        self.K = K
        self.spmm = spmm
        #self.temp = nn.Parameter(jt.ones([self.K + 1], dtype='float32'))

        self.temp= jt.random((self.K + 1,))

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.temp)
        #glorot(self.weight)
        #zeros(self.bias)
        #self._cached_adj_t = None
        #self._cached_csc=None

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        coe_tmp = nn.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K+1):
            coe[i] = coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j = math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i] = coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i] = 2*coe[i]/(self.K+1)

        Tx_0=x
        if self.spmm and jt.flags.use_cuda==1:
            Tx_1 = self.propagate_spmm(x=x, csr=csr)
        else:
            Tx_1 = self.propagate_msg(x=x, csc=csc, csr=csr)
        out=coe[0]/2*Tx_0+coe[1]*Tx_1
        for i in range(2,self.K+1):
            if self.spmm and jt.flags.use_cuda==1:
                Tx_2 = self.propagate_spmm(x=Tx_1, csr=csr)
            else:
                Tx_2 = self.propagate_msg(x=Tx_1, csc=csc, csr=csr)

            Tx_2 = 2*Tx_2-Tx_0
            out = out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2

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
