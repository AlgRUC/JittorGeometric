from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var,nn,Module
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight, cootocsc, cootocsr

def clustergcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, diag_lambda = 0.0, dtype=None):
    
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
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[row]
        edge_weight[row == col] += diag_lambda * deg_inv[row[row == col]]
        return edge_index, edge_weight


class ClusterGCNConv(MessagePassing):
    r"""The ClusterGCN graph convolutional operator from the
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper.
    
    This class implements the ClusterGCN layer, which updates node representations by aggregating information from neighboring nodes,
    adding lambda times of embedding itself, taking the graph structure into account. 
    It supports both message-passing and sparse matrix multiplication (SPMM) for propagation.

    Args:
        in_channels (int): Size of each input sample (number of input features per node).
        out_channels (int): Size of each output sample (number of output features per node).
        diag_lambda (float): Diagonal enhancement value.
        improved (bool, optional): Improved self loop value (can be ignored, similar function as diag_lambda)
        cached (bool, optional): Caching the processed csr, csc and edge_weight.
        add_self_loops (bool optional): Whether to add self-loops to the input graph.
        normalize (bool, optional): SET TO TRUE FOR NORMALLY FUNCTION.
        bias (bool, optional): If set to `True`, adds a learnable bias to the output. Default is `True`.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        **kwargs (optional): Additional arguments for the base `Module`.
    
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]
    _cached_csc: Optional[CSC]

    def __init__(self, in_channels: int, out_channels: int,
                 diag_lambda: float = 0.0,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, spmm: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(ClusterGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight_out = jt.random((in_channels, out_channels))
        self.weight_root = jt.random((in_channels, out_channels))
        
        if bias:
            self.bias = jt.random((out_channels,))
        else:
            self.bias = None
        self.spmm = spmm

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_out)
        glorot(self.weight_root)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def execute(self, x: Var, edge_index: Adj,
                edge_weight: OptVar = None) -> Var:
        """"""
        if self.normalize:
            if isinstance(edge_index, Var):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = clustergcn_norm(
                        edge_index, edge_weight, x.size(0),
                        self.improved, self.add_self_loops, self.diag_lambda)
                    with jt.no_grad():
                        csc = cootocsc(edge_index, edge_weight, x.size(0))
                        csr = cootocsr(edge_index, edge_weight, x.size(0))
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, csr, csc)
                else:
                    edge_index, edge_weight, csr, csc = cache[0], cache[1], cache[2], cache[3]
        # print("before nn")
        # abs_x=x.abs().sum()
        # print(abs_x)
        # x = x @ self.weight
        # print("after nn")
        # abs_x=x.abs().sum()
        # print(abs_x)
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
        #                      size=None)
        # print("after graph")
        # abs_x=out.abs().sum()
        # print(abs_x)
        if self.spmm and jt.flags.use_cuda==1:
            out = self.propagate_spmm(x=x, csr=csr)
        else:
            out = self.propagate_msg(x=x, csc=csc,csr=csr)
        out = out @ self.weight_out + x @ self.weight_root
        if self.bias is not None:
            out += self.bias
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
