from typing import Union, Optional, Iterable, Dict, Any  
import networkx as nx  
import jittor as jt  
from jittor import Var  
from jittor_geometric.data import Data 

def to_networkx(  
    data,  
    node_attrs: Optional[Iterable[str]] = None,  
    edge_attrs: Optional[Iterable[str]] = None,  
    graph_attrs: Optional[Iterable[str]] = None,  
    to_undirected: Optional[Union[bool, str]] = False,  
    to_multi: bool = False,  
    remove_self_loops: bool = False,  
) -> Any:  
    r"""Converts a :class:`jittor_geometric.data.Data` instance to a  
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or  
    a directed :obj:`networkx.DiGraph` otherwise.  
  
    Args:  
        data (jittor_geometric.data.Data): A homogeneous data object.  
        node_attrs (iterable of str, optional): The node attributes to be  
            copied. (default: :obj:`None`)  
        edge_attrs (iterable of str, optional): The edge attributes to be  
            copied. (default: :obj:`None`)  
        graph_attrs (iterable of str, optional): The graph attributes to be  
            copied. (default: :obj:`None`)  
        to_undirected (bool or str, optional): If set to :obj:`True`, will  
            return a :class:`networkx.Graph` instead of a  
            :class:`networkx.DiGraph`. (default: :obj:`False`)  
        to_multi (bool, optional): if set to :obj:`True`, will return a  
            :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`.  
            (default: :obj:`False`)  
        remove_self_loops (bool, optional): If set to :obj:`True`, will not  
            include self-loops in the resulting graph. (default: :obj:`False`)  
    """  
      
    to_undirected_upper: bool = to_undirected == 'upper'  
    to_undirected_lower: bool = to_undirected == 'lower'  
  
    to_undirected = to_undirected is True  
    to_undirected |= to_undirected_upper or to_undirected_lower  
    assert isinstance(to_undirected, bool)  
  
    if to_undirected:  
        G = nx.MultiGraph() if to_multi else nx.Graph()  
    else:  
        G = nx.MultiDiGraph() if to_multi else nx.DiGraph()  
  
    def to_networkx_value(value: Any) -> Any:  
        if isinstance(value, Var):  
            return value.tolist()  
        return value  
  
    # Add graph attributes  
    for key in graph_attrs or []:  
        if key in data:  
            G.graph[key] = to_networkx_value(data[key])  
  
    # Add nodes  
    num_nodes = data.num_nodes  
    if num_nodes is None:  
        num_nodes = 0  
        if data.edge_index is not None:  
            num_nodes = int(data.edge_index.max().item()) + 1  
  
    for i in range(num_nodes):  
        node_kwargs: Dict[str, Any] = {}  
        for key in node_attrs or []:  
            if key in data and data[key] is not None:  
                if isinstance(data[key], Var) and data[key].ndim > 0:  
                    node_kwargs[key] = to_networkx_value(data[key][i])  
                else:  
                    node_kwargs[key] = to_networkx_value(data[key])  
        G.add_node(i, **node_kwargs)  
  
    # Add edges  
    if data.edge_index is not None:  
        edge_list = data.edge_index.t().tolist()  
        for i, (v, w) in enumerate(edge_list):  
            if to_undirected_upper and v > w:  
                continue  
            elif to_undirected_lower and v < w:  
                continue  
            elif remove_self_loops and v == w:  
                continue  
  
            edge_kwargs: Dict[str, Any] = {}  
            for key in edge_attrs or []:  
                if key in data and data[key] is not None:  
                    if isinstance(data[key], Var) and data[key].ndim > 0:  
                        edge_kwargs[key] = to_networkx_value(data[key][i])  
                    else:  
                        edge_kwargs[key] = to_networkx_value(data[key])  
  
            G.add_edge(v, w, **edge_kwargs)  
  
    return G