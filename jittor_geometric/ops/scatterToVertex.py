'''
Description: 
Author: lusz
Date: 2024-07-05 17:22:55
'''


import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/edgetovertex_op.cc")
header = os.path.join(module_path, "cpp/edgetovertex_op.h")
srcb = os.path.join(module_path, "cpp/scattertoedge_op.cc")
headerb = os.path.join(module_path, "cpp/scattertoedge_op.h")
scatter_op = jt.compile_custom_ops((src, header))
scatter_backward_op = jt.compile_custom_ops((srcb, headerb))
# Run the test
jt.flags.use_cuda=0
class ScatterToVertexFunc(Function):
    def execute(self,x,csc,flow):
        self.flow=flow
        self.csc=csc
        # output dim
        e_num=jt.size(csc.row_indices,0)
        feature_dim=jt.size(x,1)        
        v_num=jt.size(csc.column_offset,0)-1
        self.e_num=e_num
        self.v_num=v_num
        self.feature_dim=feature_dim
        output=jt.zeros(v_num,feature_dim)
        dtype=x.dtype
        self.dtype=dtype
        scatter_op.edgetovertex(output,x,csc.row_indices,csc.column_offset,1,dtype).fetch_sync()
        return output

    def grad(self, grad_output):
        dtype=self.dtype
        output_grad=jt.zeros(self.e_num,self.feature_dim)
        csc=self.csc
        scatter_backward_op.scattertoedge(output_grad,grad_output,csc.row_indices,csc.column_offset,False,1,dtype).fetch_sync()
        return output_grad,None,None
    

def ScatterToVertex(x,csc,flow):
    out = ScatterToVertexFunc.apply(x,csc,flow)
    return out