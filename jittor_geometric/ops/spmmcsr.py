'''
Description: 
Author: lusz
Date: 2024-11-06 19:05:55
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSR
module_path = os.path.dirname(__file__)
from jittor.compile_extern import cusparse_ops
# src = os.path.join(module_path, "cpp/spmmcsr_op.cc")
# header = os.path.join(module_path, "cpp/spmmcsr_op.h")
# spmmcsr_op = jt.compile_custom_ops((src, header))
# latest jittor
# Run the test
jt.flags.use_cuda=1
class SpmmCsrFunc(Function):
    def execute(self,x,csr,trans_A,trans_B):
        self.csr=csr
        feature_dim=jt.size(x,1)        
        v_num=jt.size(csr.row_offset,0)-1
        self.v_num=v_num
        self.feature_dim=feature_dim
        output=jt.zeros(v_num,feature_dim)
        self.trans_A=trans_A
        self.trans_B=trans_B
        cusparse_ops.cusparse_spmmcsr(output,x,csr.column_indices,csr.edge_weight,csr.row_offset,v_num,v_num,trans_A,trans_B).fetch_sync()
        # spmmcsr_op.spmmcsr(output,x,csr.column_indices,csr.edge_weight,csr.row_offset,v_num,v_num).fetch_sync()
        return output

    def grad(self, grad_output):
        output_grad=jt.zeros(self.v_num,self.feature_dim)
        cusparse_ops.cusparse_spmmcsr(output_grad,grad_output,self.csr.column_indices,self.csr.edge_weight,self.csr.row_offset,self.v_num,self.v_num,self.trans_A,self.trans_B).fetch_sync()
        # spmmcsr_op.spmmcsr(output_grad,grad_output,self.csr.column_indices,self.csr.edge_weight,self.csr.row_offset,self.v_num,self.v_num).fetch_sync()
        return output_grad,None
    

def SpmmCsr(x,csr,trans_A=True,trans_B=False):
    out = SpmmCsrFunc.apply(x,csr,trans_A,trans_B)
    return out