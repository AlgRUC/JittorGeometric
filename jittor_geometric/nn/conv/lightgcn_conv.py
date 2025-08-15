'''
Description:
Author: zhengyp
Date: 2025-07-13
'''
from jittor import Var
from typing import Optional
from jittor.sparse import SparseVar, spmm
import jittor as jt
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.typing import Adj, OptVar

class LightGCNConv(MessagePassing):
    _cached_x: Optional[Var]

    def __init__(self, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(LightGCNConv, self).__init__(**kwargs)

        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
        self._cached_adj_t = None
        self._cached_csc = None

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        """Perform forward propagation."""
        if self.spmm and jt.flags.use_cuda == 1:
            x = self.propagate_spmm(x=x, csr=csr)
        else:
            x = self.propagate_msg(x=x, csc=csc, csr=csr)
        return x

    # propagate by message passing
    def propagate_msg(self, x, csc: CSC, csr: CSR):
        out = aggregateWithWeight(x, csc, csr)
        return out

    # propagate by spmm
    def propagate_spmm(self, x, csr: CSR):
        out = SpmmCsr(x, csr)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)