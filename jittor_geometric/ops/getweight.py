'''
Description:  
Author: liuyy
'''
import jittor as jt
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/getweight_op.cc")
header = os.path.join(module_path, "cpp/getweight_op.h")

getweight_op = jt.compile_custom_ops((src, header))
'''
description: Computes edge weights for a graph in CSC (Compressed Sparse Column) format using the symmetric normalization formula: weight = 1 / sqrt(out_degree[src] * in_degree[dst]).
param {*} vtx_size (int): The number of vertices in the graph.
param {*} csc_layer_dst (Var): A 1D Var containing the destination node IDs for each column in the CSC representation.
param {*} csc_layer_src (Var): A 1D Var containing the source node IDs corresponding to each edge in `row_indices`.
param {*} csc_layer_column_offset (Var): A 1D Var representing the column offsets of the CSC representation. Each element indicates the starting index of a column in `csc_layer_row_indices`.
param {*} csc_layer_row_indices (Var): A 1D Var containing the row indices (source nodes) for each edge in column-major order.
return {*}: Returns a 1D Var `csc_layer_edge_weight` containing the computed edge weights for each edge in the CSC representation.
author: liuyy
'''


def getweight(vtx_size,csc_layer_dst,csc_layer_src,csc_layer_column_offset,csc_layer_row_indices):


    csc_layer_edge_weight = jt.zeros([csc_layer_column_offset[-1].item()], dtype="float32")


    getweight_op.getweight(vtx_size,csc_layer_dst,csc_layer_src,csc_layer_column_offset,csc_layer_row_indices,csc_layer_edge_weight).fetch_sync()

    return csc_layer_edge_weight
