'''
Description: 
Author: lusz
Date: 2024-07-03 13:50:35
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/edgesoftmax_op.cc")
header = os.path.join(module_path, "cpp/edgesoftmax_op.h")
edge_softmax_op = jt.compile_custom_ops((src, header))

src_b = os.path.join(module_path, "cpp/edgesoftmaxbackward_op.cc")
header_b = os.path.join(module_path, "cpp/edgesoftmaxbackward_op.h")
edge_softmax_backward_op = jt.compile_custom_ops((src_b, header_b))


class EdgeSoftmaxFunc(Function):
    def execute(self,x,csc):
        self.x=x
        self.csc=csc
        output=jt.zeros_like(x)
        self.dtype=x.dtype
        edge_softmax_op.edgesoftmax(output,x,csc.row_indices,csc.column_offset,self.dtype)
        self.y=output
        return output

    def grad(self, grad_output):
        output_grad=jt.zeros_like(grad_output)
        edge_softmax_backward_op.edgesoftmaxbackward(output_grad,grad_output,self.y,self.csc.row_indices,self.csc.column_offset,self.dtype)
        return output_grad,None
    


def EdgeSoftmax(x,csc):
    out = EdgeSoftmaxFunc.apply(x,csc)
    return out