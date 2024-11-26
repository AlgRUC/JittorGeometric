'''
Description: COO version of scatter to edge for GAT
Author: lusz
Date: 2024-06-28 17:10:12
'''

import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/scattertoedge_op.cc")
header = os.path.join(module_path, "cpp/scattertoedge_op.h")
srcb = os.path.join(module_path, "cpp/edgetovertex_op.cc")
headerb = os.path.join(module_path, "cpp/edgetovertex_op.h")
scatter_op = jt.compile_custom_ops((src, header))
scatter_backward_op = jt.compile_custom_ops((srcb, headerb))
# Run the test
jt.flags.use_cuda=0
class ScatterToEdgeFunc(Function):
    def execute(self,x,csc,flow):
        self.flow=flow
        self.csc=csc
        # output dim
        e_num=jt.size(csc.row_indices,0)
        feature_dim=jt.size(x,1)        
        v_num=jt.size(x,0)
        self.e_num=e_num
        self.v_num=v_num
        self.feature_dim=feature_dim
        output=jt.zeros(e_num,feature_dim)
        dtype=x.dtype
        self.dtype=dtype
        flag=1
        if flow=="src":
            flag=0
        self.flag=flag
        scatter_op.scattertoedge(output,x,csc.row_indices,csc.column_offset,False,flag,dtype).fetch_sync()
        
        return output

    def grad(self, grad_output):
        dtype=self.dtype
        output_grad=jt.zeros(self.v_num,self.feature_dim)
        csc=self.csc
        scatter_backward_op.edgetovertex(output_grad,grad_output,csc.row_indices,csc.column_offset,self.flag,dtype).fetch_sync()
        return output_grad,None,None
    

def ScatterToEdge(x,csc,flow):
    out = ScatterToEdgeFunc.apply(x,csc,flow)
    return out