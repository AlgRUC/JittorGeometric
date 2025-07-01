'''
Description: Converts the graph to an undirected graph 
Author: lusz
Date: 2024-06-23 14:45:47
'''
import jittor as jt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/toundirected_op.cc")
header = os.path.join(module_path, "cpp/toundirected_op.h")
toundirected_op = jt.compile_custom_ops((src, header))

def toUndirected(edge_index, edge_attr,num_nodes):
    num_edges=jt.size(edge_index,1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    new_edge_index=jt.zeros_like(edge_index)
    new_edge_attr=jt.zeros_like(edge_attr)
    toundirected_op.toundirected(edge_index,edge_attr,num_edges,num_nodes,new_edge_index,new_edge_attr,edge_attr.dtype)
    return new_edge_index,new_edge_attr
