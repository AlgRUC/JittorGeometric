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

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None] #聚合方法 ('add', 'mean', 'max')。默认为 'add'

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source'] # 信息传递的方向 ('source_to_target' 或 'target_to_source')

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.scatter_to_edge)
        self.inspector.inspect(self.edge_forward,pop_first=True)
        
        # print(self.special_args) {'size', 'edge_index', 'index', 'ptr', 'edge_index_j', 'edge_index_i', 'size_i', 'dim_size', 'adj_t', 'size_j'}
        self.__user_args__ = self.inspector.keys(
            ['scatter_to_edge','edge_forward']).difference(self.special_args) # difference返回两个集合之间的差集，即存在于第一个集合但不存在于第二个集合的元素
        # print(self.__user_args__) {'edge_weight', 'x_j'}

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Var):
            assert edge_index.dtype == Var.int32
            assert edge_index.ndim == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `jittor Var int32` of '
             'shape `[2, num_messages]`'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Var):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered Var with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Var):
            index = edge_index[dim]
            return src[(slice(None),)*self.node_dim+(index,)]
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            # print(arg) edge_weight x_j
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty)
                if isinstance(data, (tuple, list)):
                    # 如果数据是元组或列表
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Var):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Var):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Var):
            # print("isinstance") # 每个epoch调用了4次
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self,x, csc: CSC, csr:CSR,size: Size = None):
        # x = kwargs.get('x')
        # print("x")
        # abs_x=x.abs().sum()
        # print(abs_x)
        out = self.aggregate_with_weight(x,csc,csr)
        # print("out")
        # abs_out=out.abs().sum()
        # print(abs_out)
        
        return out

    def aggregate_with_weight(self,x,csc,csr)->Var:
        """
        Used for GCN demo ,combine  'scatter_to_edge' with  'scatter_to_vertex'
        """
        output=aggregateWithWeight(x,csc,csr)
        return output

    def scatter_to_edge(self,x,csc)->Var:
        """
        ScatterToEdge is an edge message generating operation t
        hat scatters the source and destination representations 
        to edges for the EdgeForward computation
        """
        return
    
    def edge_forward(self,x_j:Var)->Var:
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
    
    def vertex_forward(self,x_j:Var)->Var:
        """
        VertexForward is a parameterized function defined on each vertex 
        to generate new vertex representation by applying zero or several 
        NN models on aggregated neighborhood representations.
        """
        return
