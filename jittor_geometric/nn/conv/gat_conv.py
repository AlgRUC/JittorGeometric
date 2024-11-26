'''
Description: 
Author: lusz
Date: 2024-06-26 10:57:06
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassingNts
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC,CSR
from jittor_geometric.ops import ScatterToEdge,EdgeSoftmax,aggregateWithWeight,ScatterToVertex


class GATConv(MessagePassingNts):
    r"""The graph convolutional operator from the `"Graph Attention Networks"
    2018 ICLR _ paper
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]
    _cached_csc: Optional[CSC]

    def __init__(self, in_channels: int, out_channels: int,e_num: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = jt.random((in_channels, out_channels))
        self.edge_weight=jt.random((2*out_channels,1))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        self._cached_adj_t = None
        self._cached_csc=None

    def execute(self, x: Var, csc: CSC) -> Var:
        """"""
        out=self.vertex_forward(x)
        out = self.propagate(x=out,csc=csc)
        return out
    
    def propagate(self,x,csc):
        e_msg=self.scatter_to_edge(x,csc)
        out = self.edge_forward(e_msg,csc)
        out=self.scatter_to_vertex(out,csc)
        return out
    
    def scatter_to_edge(self,x,csc)->Var:
        out1=ScatterToEdge(x,csc,"src")
        out2=ScatterToEdge(x,csc,"dst")
        out =jt.contrib.concat([out1,out2],dim=1)
        return out
    
    def edge_forward(self,x,csc)->Var:
        out = x @ self.edge_weight
        m=jt.nn.leaky_relu(out,scale=0.2)
        a=EdgeSoftmax(m,csc)
        half_dim=int(jt.size(x,1)/2)
        e_msg=x[:,0:half_dim]
        return e_msg * a
    
    def scatter_to_vertex(self,edge,csc)->Var:
        out=ScatterToVertex(edge,csc,'src')
        return out
    
    def vertex_forward(self,x:Var)->Var:
        out = x @ self.weight
        out=jt.nn.relu(out)
        return out
    

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
