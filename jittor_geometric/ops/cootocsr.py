'''
Description: Convert COO to CSR
Author: lusz
Date: 2024-06-21 19:40:07
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSR
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/cootocsr_op.cc")
header = os.path.join(module_path, "cpp/cootocsr_op.h")

cootocsr_op = jt.compile_custom_ops((src, header))


'''
description: Converts a graph from COO (Coordinate) format to CSR (Compressed Sparse Row) format.
param {*} edge_index(Var): The indices of the edges in the COO format. It is expected to be a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.
param {*} edge_weight(Var): The weights of the edges in the COO format. It is a 1D Var where each element represents the weight of the corresponding edge.
param {*} v_num(int): The number of vertices in the graph.
return {*}: Returns a CSR representation of the graph, which includes column indices, row offsets, and edge weights.
author: lusz
'''
def cootocsr(edge_index,edge_weight,v_num):
    e_num=jt.size(edge_weight,0)
    csr_edge_weight=jt.zeros(e_num)
    column_indices = jt.zeros((e_num,), dtype='int32')
    row_offset = jt.zeros((v_num+1,), dtype='int32')
    dtype=edge_weight.dtype
    cootocsr_op.cootocsr(edge_index, edge_weight, column_indices, row_offset, csr_edge_weight, v_num, dtype).fetch_sync()
    csr=CSR(column_indices,row_offset,csr_edge_weight)
    return csr