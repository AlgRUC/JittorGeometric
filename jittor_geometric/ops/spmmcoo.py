'''
Description: 
Author: lusz
Date: 2024-11-11 14:10:31
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/spmmcoo_op.cc")
header = os.path.join(module_path, "cpp/spmmcoo_op.h")
print(src)
print(header)
spmmcoo_op = jt.compile_custom_ops((src, header))
# Run the test
jt.flags.use_cuda=1
class SpmmCooFunc(Function):
    def execute(self,x,edge_index,edge_weight):
        print(x)
        self.edge_index=edge_index
        row_indices=edge_index[0,:]
        col_indices=edge_index[1,:]
        print(row_indices)
        print(col_indices)
        print(edge_weight)
        self.row_indices=row_indices
        self.col_indices=col_indices
        self.edge_weight=edge_weight
        feature_dim=jt.size(x,1)        
        v_num=jt.size(x,0)
        self.v_num=v_num
        self.feature_dim=feature_dim
        output=jt.zeros(v_num,feature_dim)
        dtype=x.dtype
        self.dtype=dtype
        spmmcoo_op.spmmcoo(output,x,row_indices,col_indices,edge_weight,v_num,v_num,dtype).fetch_sync()
        print(output)
        return output

    def grad(self, grad_output):
        output_grad=jt.zeros(self.v_num,self.feature_dim)
        spmmcoo_op.spmmcoo(output_grad,grad_output,self.row_indices,self.col_indices,self.edge_weight,self.v_num,self.v_num,self.dtype).fetch_sync()
        return output_grad,None,None
    

def SpmmCoo(x,edge_index,edge_weight):
    out = SpmmCooFunc.apply(x,edge_index,edge_weight)
    return out