'''
Author: lusz
Date: 2025-04-26 15:44:53
Description: 
'''
import jittor as jt
import os
import sys
from jittor import nn,Var
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/indexselect_op.cc")
header = os.path.join(module_path, "cpp/indexselect_op.h")
src_b = os.path.join(module_path, "cpp/indexselectbwd_op.cc")
header_b = os.path.join(module_path, "cpp/indexselectbwd_op.h")
indexselect_op = jt.compile_custom_ops((src, header))
indexselectbwd_op = jt.compile_custom_ops((src_b, header_b))
# Run the test
class IndexselectFunc(Function):
    def execute(self,x,dim,index):
        self.dim=dim
        self.v_num=jt.size(x,0)
        self.feature_dim=jt.size(x,1)
        self.index=index
        i_num=jt.size(index,0)
        embedding_dim=jt.size(x,0)
        output=jt.zeros(i_num,embedding_dim)
        indexselect_op.indexselect(output,x,dim,index).fetch_sync()
        return output

    def grad(self, grad_output):
        output_grad=jt.zeros(self.v_num,self.feature_dim)
        indexselectbwd_op.indexselectbwd(output_grad,grad_output,self.dim,self.index).fetch_sync()
        return output_grad,None,None
    

def IndexSelect(x,dim,index):
    out = IndexselectFunc.apply(x,dim,index)
    return out