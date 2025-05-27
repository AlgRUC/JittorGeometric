'''
Description: Convert COO to CSC
Author: lusz
Date: 2024-06-21 20:20:48
'''

import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/cootocsc_op.cc")
header = os.path.join(module_path, "cpp/cootocsc_op.h")

cootocsc_op = jt.compile_custom_ops((src, header))


'''
description: Converts a graph from COO (Coordinate) format to CSC (Compressed Sparse Row) format.
param {*} edge_index(Var): The indices of the edges in the COO format. It is expected to be a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.
param {*} edge_weight(Var): The weights of the edges in the COO format. It is a 1D Var where each element represents the weight of the corresponding edge.
param {*} v_num(int): The number of vertices in the graph.
return {*}: Returns a CSC representation of the graph, which includes column indices, row offsets, and edge weights.
author: lusz
'''
def cootocsc(edge_index,edge_weight,v_num):
    e_num=jt.size(edge_weight,0)
    csc_edge_weight=jt.zeros(e_num)
    row_indices = jt.zeros((e_num,), dtype='int32')
    column_offset = jt.zeros((v_num+1,), dtype='int32')
    cootocsc_op.cootocsc(edge_index, edge_weight, row_indices, column_offset, csc_edge_weight, v_num).fetch_sync()
    csc=CSC(row_indices,column_offset,csc_edge_weight)
    return csc
