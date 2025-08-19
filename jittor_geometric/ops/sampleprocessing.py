'''
Description: Converts the graph to an undirected graph 
Author: lusz
Date: 2024-06-23 14:45:47
'''
import jittor as jt
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/sampleprocessing_op.cc")
header = os.path.join(module_path, "cpp/sampleprocessing_op.h")

sampleprocessing_op = jt.compile_custom_ops((src, header))
'''
description: Performs neighbor sampling for a layer by processing the global CSC representation into a local CSC representation, respecting the specified fanout.
param {*} fanout_i (int): The maximum number of neighbors to sample for each destination node.
param {*} csc_layer_dst (Var): A 1D Var containing the IDs of the destination nodes for the current layer.
param {*} csc_layer_column_offset (Var): A 1D Var to store the local CSC column offsets for the sampled neighbors; its size should be `len(csc_layer_dst) + 1`.
param {*} csc_global_column_offset (Var): The global CSC column offset array indicating the start index of each node in `csc_global_row_indices`.
param {*} csc_global_row_indices (Var): The global CSC row indices array representing source nodes of all edges.
return {*}: Returns a 1D Var `csc_layer_row_indices` containing the source node indices of the sampled neighbors in column-major order, matching `csc_layer_column_offset`.
author: liuyy
'''

def sampleprocessing(fanout_i,csc_layer_dst,csc_layer_column_offset,csc_global_column_offset,csc_global_row_indices):

    csc_layer_row_indices = jt.zeros([csc_layer_column_offset[-1].item()], dtype="int32")
   
    sampleprocessing_op.sampleprocessing(fanout_i,csc_layer_dst,csc_layer_column_offset,csc_layer_row_indices,csc_global_column_offset,csc_global_row_indices)
 
    return csc_layer_row_indices
