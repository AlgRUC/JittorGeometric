'''
Description: 
Author: lusz
Date: 2024-06-23 14:53:18
'''
import jittor as jt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.ops import toUndirected

jt.flags.lazy_execution = 0
edge_index = jt.array([[0, 1, 1],
                       [2, 0, 2]])
edge_attr = jt.array([1., 3., 2.])
num_edges=3
num_nodes=3
new_edge_index,new_edge_attr=toUndirected(edge_index,edge_attr,num_nodes)
print(new_edge_index)
print(new_edge_attr)