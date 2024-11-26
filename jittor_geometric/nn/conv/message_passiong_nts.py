import os
import os.path as osp
import sys
from typing import List, Optional, Set
from inspect import Parameter

import jittor as jt
from jittor import nn, Module
from jittor import Var
from jittor_geometric.typing import Adj, Size

from .utils.inspector import Inspector
from jittor_geometric.data import CSC,CSR
from jittor_geometric.ops import aggregateWithWeight

class MessagePassingNts(Module):
    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):

        super(MessagePassingNts, self).__init__()

    # graph operations
    def propagate(self,x):
        return 

    def aggregate_with_weight(self,x,csc,csr)->Var:
        """
        Used for GCN demo ,combine  'scatter_to_edge' with  'scatter_to_vertex'
        """
        output=aggregateWithWeight(x,csc,csr)
        return output

    def scatter_to_edge(self,x)->Var:
        """
        ScatterToEdge is an edge message generating operation t
        hat scatters the source and destination representations 
        to edges for the EdgeForward computation
        """
        return
    
    def edge_forward(self,x:Var)->Var:
        """
        EdgeForward is a parameterized function defined on each 
        edge to generate an output message by combining the edge 
        representation with the representations of source and destination.
        """
        return
    
    def scatter_to_vertex(self,x,csc)->Var:
        """
        Scatter_to_vertex takes incoming edge-associated Vars as input 
        and outputs a new vertex representation for next layer's computation
        """
        return
    
    def vertex_forward(self,x:Var)->Var:
        """
        VertexForward is a parameterized function defined on each vertex 
        to generate new vertex representation by applying zero or several 
        NN models on aggregated neighborhood representations.
        """
        return
