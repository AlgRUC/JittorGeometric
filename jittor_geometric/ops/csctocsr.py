'''
Description: Converts CSC to CSR
Author: liuyy
'''
import jittor as jt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes

module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/csctocsr_op.cc")
header = os.path.join(module_path, "cpp/csctocsr_op.h")
csctocsr_op = jt.compile_custom_ops((src, header))
'''
description: Converts a graph from CSC (Compressed Sparse Column) format to CSR (Compressed Sparse Row) format.
param {*} column_offset(Var): The column offset array of the CSC representation. It is a 1D Var where each element indicates the starting index of a column in `row_indices`.
param {*} row_indices(Var): The row indices array of the CSC representation. It is a 1D Var containing the source node indices for each edge in column-major order.
param {*} dst(Var): A 1D Var containing the destination node IDs in the CSR representation.
param {*} src(Var): A 1D Var containing the source node IDs in the CSR representation.
return {*}: Returns a tuple `(row_offset, column_indices)` representing the CSR format of the graph. `row_offset` is a 1D Var where each element indicates the starting index of a row in `column_indices`. `column_indices` is a 1D Var containing the destination node indices for each edge in row-major order.
author: liuyy
'''


def csctocsr(column_offset, row_indices,dst,src):
    #print("def csctocsr开启\n")
    row_offset = jt.zeros([jt.size(src,0) + 1], dtype="int32")
    column_indices = jt.zeros([column_offset[-1].item()], dtype="int32")


    csctocsr_op.csctocsr(column_offset,row_indices,row_offset,column_indices,dst,src)
    return row_offset,column_indices
