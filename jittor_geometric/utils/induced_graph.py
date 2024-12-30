import jittor
from typing import Tuple

def induced_graph(
    edge_index: jittor.Var,
    node_selected: jittor.Var,
    max_nodes: int
) -> Tuple[jittor.Var, jittor.Var]:
    r"""Generate the node-induced graph of the original graph described by
    'edge_index'. It is expected that max_nodes is larger than, or equal to
    the real max node index in edge_index, or may cause errors.
    
    Args:
        edge_index (jittor.Var): The edge list describing the whole graph.
        node_selected (jittor.Var): The node list of the expected induced 
            graph.
        max_nodes (int): The maximum node index in edge_index.
    """
    node_mask = jittor.zeros(max_nodes, dtype = "bool")
    node_mask[node_selected] = True
    node_map = jittor.zeros(max_nodes, dtype = "int")
    node_map[node_selected] = jittor.arange(0, node_selected.size(0))
    
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_selected = jittor.nonzero(edge_mask).view(-1)
    
    return node_map, edge_selected