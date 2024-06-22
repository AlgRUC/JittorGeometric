from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassingNts
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
# add by lusz
from jittor_geometric.data import CSC,CSR
from jittor_geometric.ops import addone

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, Var):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = jt.ones((edge_index.size(1), ))

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        shape = list(edge_weight.shape)
        shape[0] = num_nodes
        deg = jt.zeros(shape)
        deg = jt.scatter(deg, 0, col, src=edge_weight, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConvNts(MessagePassingNts):
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
        super(GCNConvNts, self).__init__(**kwargs)

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
        # self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_csc=None

    def execute(self, x: Var, csc: OptVar, csr: OptVar) -> Var:
        """"""

        if self.normalize:
            cache = self._cached_csc
            if cache is None:
                if self.cached:
                    self._cached_csc = csc       
            else:
                csc = cache
        
        # print("before nn")
        # abs_x=x.abs().sum()
        # print(abs_x)
        x = x @ self.weight
        # print("after nn")
        # abs_x=x.abs().sum()
        # print(abs_x)
        out = self.propagate(x=x, csc=csc,csr=csr,size=None)
        # abs_x=out.abs().sum()
        # print("after graph")
        # print(abs_x)
        if self.bias is not None:
            out += self.bias
    
        return out

    def scatter_to_edge(self, x, csc):
        v_num=jt.size(x,0)
        feature_dim=jt.size(x,1)
        # print(x.shape)
        # result=jt.zeros((v_num,feature_dim))
        result=addone(x,1,v_num*feature_dim)
        # my_op = jt.compile_custom_op(header_add, src_add, "addone", warp=False)
        # my_op(result,x,2.0,v_num*feature_dim,'float32').fetch_sync() 
        # edge_num=jt.size(csc.edge_weight,0)
        # feature_dim=jt.size(x,1)
        # # print(type(feature_dim))
        # result = jt.zeros((edge_num,feature_dim))
        # my_op = jt.compile_custom_op(header_col, src_col, "colassign", warp=False)
        # for i in range(edge_num):
        #     # print(i)
        #     row=int(csc.row_indices[i])
        #     my_op(result,x,i,row,feature_dim,'float32').fetch_sync() 

        #     # print(2)
        # # for col in range(len(csc.column_offset) - 1):
        # #     start = csc.column_offset[col]
        # #     end = csc.column_offset[col + 1]
        # #     for i in range(start, end):
        # #         row = csc.row_indices[i]
        # #         result[row] = x[col]
        
        # return result
    
    # def scatter_to_vertex(self,x,csc):
    #     num_vertices = jt.size(csc.column_offset,0) - 1
    #     vertex_features = jt.zeros((num_vertices, jt.size(x,1)))
    #     for col in range(num_vertices):
    #         start = csc.column_offset[col]
    #         end = csc.column_offset[col + 1]
    #         for i in range(start, end):
    #             vertex_features[col] += x[i] * csc.edge_weight[i]
    #     # print(jt.size(vertex_features))
    #     return vertex_features
        
    
    # def edge_forward(self,x_j:Var)->Var:
    #     return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
