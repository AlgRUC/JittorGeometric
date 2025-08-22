'''
Description: 
Author: lusz
Date: 2024-11-29 10:14:19
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar
import jittor as jt
from jittor import Var,nn,Module
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
if not jt.compiler.has_acl: 
    from jittor_geometric.ops import SpmmCsr
from jittor_geometric.ops import aggregateWithWeight
from jittor_geometric.ops import aggregateWithWeightACL

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    if isinstance(edge_index, Var):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if edge_weight is None:
            edge_weight = jt.ones((edge_index.size(1), ))
        # if add_self_loops:
        #     edge_index, tmp_edge_weight = add_remaining_self_loops(
        #         edge_index, edge_weight, fill_value, num_nodes)
        #     assert tmp_edge_weight is not None
        #     edge_weight = tmp_edge_weight
        row, col = edge_index[0], edge_index[1]
        shape = list(edge_weight.shape)
        shape[0] = num_nodes
        deg = jt.zeros(shape)
        deg = jt.scatter(deg, 0, col, src=edge_weight, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    
class GCNConv(Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    This class implements the Graph Convolutional Network (GCN) layer, which updates node representations by aggregating information from neighboring nodes, taking the graph structure into account. 
    It supports both message-passing and sparse matrix multiplication (SPMM) for propagation.

    Args:
        in_channels (int): Size of each input sample (number of input features per node).
        out_channels (int): Size of each output sample (number of output features per node).
        bias (bool, optional): If set to `True`, adds a learnable bias to the output. Default is `True`.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        **kwargs (optional): Additional arguments for the base `Module`.

    Example:
        >>> dataset = MyGraphDataset(root='/path/to/dataset')
        >>> data = dataset[0]  # Access the first graph object
        >>> conv = GCNConv(in_channels=16, out_channels=32, bias=True, spmm=True)
        >>> v_num = data.x.shape[0]  # Number of nodes
        >>> edge_index, edge_weight = data.edge_index, data.edge_attr
        >>> edge_index, edge_weight = gcn_norm(edge_index, edge_weight, v_num,
        ...                                    improved=False, add_self_loops=True)
        >>> with jt.no_grad():
        ...     csc = cootocsc(edge_index, edge_weight, v_num)
        ...     csr = cootocsr(edge_index, edge_weight, v_num)
        >>> x = jt.random((v_num, 16))  # Randomly initialize node features
        >>> out = conv(x, csc, csr)  # Apply GCN layer
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]
    _cached_csc: Optional[CSC]
    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, spmm:bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.weight = jt.random((in_channels, out_channels))
        if bias:
            self.bias = jt.random((out_channels,))
        else:
            self.bias = None
        self.spmm=spmm
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_csc=None

    def execute(self, x: Var, csc: OptVar, csr: OptVar,A) -> Var:
        x = x @ self.weight
        if self.spmm and jt.flags.use_cuda==1:
            out = self.propagate_spmm(x=x, csr=csr)
        if jt.flags.use_acl ==0:
            out = self.propagate_msg(x=x, csc=csc,csr=csr)
        if jt.flags.use_acl ==1:   
            out = jt.matmul(A,x)
            # out = self.propagate_msg_acl(x=x, csc=csc,csr=csr)
        if self.bias is not None:
            out += self.bias
        return out

    # propagate by message passing
    def propagate_msg(self,x, csc: CSC, csr:CSR):
        out = aggregateWithWeight(x,csc,csr)  
        return out
    
    def propagate_msg_acl(self,x, csc: CSC, csr:CSR):
        out = aggregateWithWeightACL(x,csc,csr) 
        return out
    # propagate by spmm
    def propagate_spmm(self, x, csr:CSR):
        out = SpmmCsr(x,csr)  
        return out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
