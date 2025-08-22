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
src = os.path.join(module_path, "cpp/gpuinitco_op.cc")
header = os.path.join(module_path, "cpp/gpuinitco_op.h")

gpuinitco_op = jt.compile_custom_ops((src, header))
'''
description: Initializes the local CSC column offset for a set of destination nodes, respecting a maximum fanout.
param {*} dst_size (int): The number of destination nodes.
param {*} fanout_i (int): The maximum number of neighbors to sample for each destination node.
param {*} csc_layer_dst (Var): A 1D Var containing the IDs of the destination nodes.
param {*} csc_global_column_offset (Var): The global CSC column offset array, indicating the start index of each node in the global row_indices array.
return {*}: Returns a 1D Var `csc_layer_column_offset` of size `dst_size + 1`, where for each node i, `csc_layer_column_offset[i+1] = min(number_of_neighbors_of_node_i, fanout_i)`.
author: liuyy
'''


def gpuinitco(dst_size,fanout_i,csc_layer_dst,csc_global_column_offset):
    

    csc_layer_column_offset = jt.zeros(dst_size+1, dtype="int32")


    gpuinitco_op.gpuinitco(dst_size,fanout_i,csc_layer_dst,csc_layer_column_offset,csc_global_column_offset).fetch_sync()

    return csc_layer_column_offset
