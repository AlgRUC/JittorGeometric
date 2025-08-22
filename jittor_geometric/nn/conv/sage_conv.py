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



def sage_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
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
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill(deg_inv == float('inf'), 0)
        return edge_index, edge_weight * deg_inv[col]


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive 
    Representation Learning on Large Graphs" 
    <https://arxiv.org/abs/1706.02216>`_ paper.
    
    This class implements the GraphSAGE layer, which updates node representations by aggregating information from neighboring nodes,
    concatenating the transformed embedding itself, taking the graph structure into account. 
    It supports both message-passing and sparse matrix multiplication (SPMM) for propagation.

    Args:
        in_channels (int): Size of each input sample (number of input features per node).
        out_channels (int): Size of each output sample (number of output features per node).
        improved (bool, optional): Improved self loop value (can be ignored, similar function as diag_lambda)
        cached (bool, optional): Caching the processed csr, csc and edge_weight.
        add_self_loops (bool optional): Whether to add self-loops to the input graph.
        normalize (bool, optional): SET TO TRUE FOR NORMALLY FUNCTION.
        project (bool, optional): If set to 'True', the node feature will be transformed first.
        root_weight (bool, optional): If set to 'True', adding the transformed node features to result.
        bias (bool, optional): If set to `True`, adds a learnable bias to the output. Default is `True`.
        spmm (bool, optional): If set to `True`, uses sparse matrix multiplication (SPMM) for propagation. Default is `True`.
        **kwargs (optional): Additional arguments for the base `Module`.
    
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, 
                 normalize: bool = True,
                 project: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, spmm: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.project = project

        self._cached_edge_index = None
        self._cached_adj_t = None

        if project:
            self.weight_proj = jt.random((in_channels, in_channels))
            self.bias_proj = jt.random((in_channels, ))
        else:
            self.weight_proj = None
            self.bias_proj = None
        
        self.weight1 = jt.random((in_channels, out_channels))
        if root_weight:
            self.root_weight = jt.random((in_channels, out_channels))
        else:
            self.root_weight = None

        if bias:
            self.bias = jt.random((out_channels,))
        else:
            self.bias = None


        self.spmm = spmm
        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            glorot(self.weight_proj)
            zeros(self.bias_proj)
        glorot(self.weight1)
        if self.root_weight is not None:
            glorot(self.root_weight)
        if self.bias is not None:
            zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None
    '''
    def execute(self, x: Var, edge_index: Adj,
                edge_weight: OptVar = None) -> Var:
        """"""
        if self.normalize:
            if isinstance(edge_index, Var):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = sage_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    with jt.no_grad():
                        csc = cootocsc(edge_index, edge_weight, x.size(0))
                        csr = cootocsr(edge_index, edge_weight, x.size(0))
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, csr, csc)
                else:
                    edge_index, edge_weight, csr, csc = cache[0], cache[1], cache[2], cache[3]
        
        x = (x, x)
        if self.weight_proj is not None:
            x = (jt.nn.relu(x[0] @ self.weight_proj + self.bias_proj), x[1])
        
        if self.spmm and jt.flags.use_cuda==1:
            out = self.propagate_spmm(x=x[0], csr=csr)
        else:
            out = self.propagate_msg(x=x[0], csc=csc,csr=csr)
        out = out @ self.weight1
         
        if self.root_weight is not None:
            out = out + x[1] @ self.root_weight
       
        if self.bias is not None:
            out += self.bias
        return out
    '''

    def execute(self, x: Var, edge_index: Adj = None,
            edge_weight: Var = None,
            csc = None, csr = None) -> Var:
        """
        Args:
            x:  (Var)
            edge_index: COO  (2, E)
            edge_weight:             
            csc: CSC
            csr: CSR
        """
        # csc/csr
        if csc is not None and csr is not None:
            
            x = (x, x)
            if self.weight_proj is not None:
                x = (jt.nn.relu(x[0] @ self.weight_proj + self.bias_proj), x[1])
            if self.spmm and jt.flags.use_cuda == 1:
                out = self.propagate_spmm(x=x[0], csr=csr)
            else:
                out = self.propagate_msg(x=x[0], csc=csc, csr=csr)

        
            out = out @ self.weight1

            if self.root_weight is not None:
                out = out + x[1] @ self.root_weight

            if self.bias is not None:
                out += self.bias
        else:
           
            if self.normalize and isinstance(edge_index, Var):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = sage_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    with jt.no_grad():
                        csc = cootocsc(edge_index, edge_weight, x.size(0))
                        csr = cootocsr(edge_index, edge_weight, x.size(0))
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, csr, csc)
                else:
                    edge_index, edge_weight, csr, csc = cache
            x = (x, x)
            if self.weight_proj is not None:
                x = (jt.nn.relu(x[0] @ self.weight_proj + self.bias_proj), x[1])


            if self.spmm and jt.flags.use_cuda == 1:
                out = self.propagate_spmm(x=x[0], csr=csr)
            else:
                out = self.propagate_msg(x=x[0], csc=csc, csr=csr)

        
            out = out @ self.weight1

            if self.root_weight is not None:
                out = out + x[1] @ self.root_weight

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
