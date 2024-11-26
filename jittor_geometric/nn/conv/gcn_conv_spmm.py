'''
Description: 
Author: lusz
Date: 2024-06-17 10:31:19
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var,nn,Module
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC,CSR
from jittor_geometric.ops import SpmmCsr
import time


class GCNConvSpmm(Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]
    _cached_csc: Optional[CSC]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConvSpmm, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = jt.random((in_channels, out_channels))

        if bias:
            self.bias = jt.random((out_channels,))

        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_csc=None

    def execute(self, x: Var, csr: OptVar) -> Var:
        x = x @ self.weight
        out = self.propagate(x=x,csr=csr)
        if self.bias is not None:
            out += self.bias
    
        return out

    def propagate(self,x, csr:CSR):
        out = SpmmCsr(x,csr)  
        return out
    

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
