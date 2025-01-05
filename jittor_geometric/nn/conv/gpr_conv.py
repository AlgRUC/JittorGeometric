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

class GPRGNN(Module):
    r"""The graph propagation operator from the `"Adaptive Universal 
    Generalized PageRank Graph Neural Network"
    <https://arxiv.org/abs/2006.07988>`_ paper

    Mathematical Formulation:
    .. math::
        \mathbf{Z} = \sum_{k=0}^{K} \alpha_k \mathbf{P}^{k} \mathbf{X}.

    Args:
        K (int): Order of polynomial, or maximum number of hops considered for message passing. 
        alpha (float): Parameter controlling the weighting of different hops.
        Init (str): Initialization method for the propagation weights. Possible values are 'SGC', 'PPR', 'NPPR', 'Random', 'WS'.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
    """

    def __init__(self, K: int, alpha: float, Init: str, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GPRGNN, self).__init__(**kwargs)
        self.K = K

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        TEMP_jt = jt.array(TEMP)
        self.temp = nn.Parameter(TEMP_jt)

        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
    	pass

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        out = x*(self.temp[0])
        for k in range(self.K):
            if self.spmm and jt.flags.use_cuda==1:
                x = self.propagate_spmm(x=x, csr=csr)
            else:
                x = self.propagate_msg(x=x, csc=csc, csr=csr)
            out = out + self.temp[k+1]*x
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
