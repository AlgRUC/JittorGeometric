'''
Description: 
Author: lusz
Date: 2024-06-21 14:50:39
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
module_path = os.path.dirname(__file__)
# print(module_path)
src = os.path.join(module_path, "aggregate_op.cc")
header = os.path.join(module_path, "aggregate_op.h")

aggregate_op = jt.compile_custom_ops((src, header))
# Run the test
class AggregateFunc(Function):
    def execute(self,x,csc,csr):
        self.csc=csc
        self.csr=csr
        edge_weight=csc.edge_weight
        indices=csc.row_indices
        offset=csc.column_offset
        dtype=edge_weight.dtype
        output=x
        aggregate_op.aggregate(output,x,indices,offset,edge_weight,True,dtype).fetch_sync()
        return output

    def grad(self, grad_output):
        edge_weight=self.csr.edge_weight
        indices=self.csr.column_indices
        offset=self.csr.row_offset
        dtype=edge_weight.dtype
        output_grad=grad_output
        aggregate_op.aggregate(output_grad,grad_output,indices,offset,edge_weight,False,dtype).fetch_sync()
        # print(output_grad)
        return output_grad,None,None
    
'''
description:  This function performs aggregation on the vertex embedding matrix using CSC (Compressed Sparse Column) 
and CSR (Compressed Sparse Row) representations of the graph
param {*} x The vertex embedding matrix of shape (v_num, dim), where `v_num` is the number of vertices and `dim` is the dimension of the embeddings.
param {*} csc
param {*} csr
return {*}
author: xuchaoxin
'''
def aggregateWithWeight(x,csc,csr):
    out = AggregateFunc.apply(x,csc,csr)
    return out