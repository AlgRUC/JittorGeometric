import time
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSR
module_path = os.path.dirname(__file__)

src = os.path.join(module_path, "cpp/distspmm_op.cc")
header = os.path.join(module_path, "cpp/distspmm_op.h")
distspmm_op = jt.compile_custom_ops((src, header))

jt.flags.use_cuda=1
class DistSpmmFunc(Function):
    def execute(self,dist,x,csr_list,local_vnum,trans_A=True,trans_B=False):

        self.dist = dist
        self.csr_list=csr_list
        feature_dim=jt.size(x,1)
        self.local_vnum = local_vnum
        self.v_num=None
        self.feature_dim=feature_dim
        self.trans_A=trans_A
        self.trans_B=trans_B

        feature_bcast = None
        result = jt.zeros_like(x)
        output = jt.zeros_like(x)
        for src in range(dist.size):
            if dist.rank == src:
                feature_bcast = x
            feature_bcast = dist.broadcast(x, src)
            
            output = jt.zeros_like(x)
            v_num=jt.size(csr_list[src].row_offset,0)-1 
            distspmm_op.distspmm(output,feature_bcast,csr_list[src].column_indices,csr_list[src].edge_weight,csr_list[src].row_offset,v_num,local_vnum,trans_A,trans_B).fetch_sync()
            result = result + output
        return result 

    def grad(self, grad_output):
        assert (grad_output.size()[0] == self.local_vnum)
        result = jt.zeros(grad_output.size()[0],self.feature_dim)
        output = jt.zeros(grad_output.size()[0],self.feature_dim)

        feature_bcast = None
        for src in range(self.dist.size):
            v_num=jt.size(self.csr_list[src].row_offset,0)-1
            if self.dist.rank == src:
                feature_bcast = grad_output
            feature_bcast = self.dist.broadcast(grad_output, src)
            
            distspmm_op.distspmm(output,feature_bcast,self.csr_list[src].column_indices,self.csr_list[src].edge_weight,self.csr_list[src].row_offset,v_num,self.local_vnum,self.trans_A, self.trans_B)

            result = result + output
        return None, result
    
def DistSpmm(dist,x,csr_list,local_vnum,trans_A=True,trans_B=False):
    out = DistSpmmFunc.apply(dist,x,csr_list,local_vnum,trans_A,trans_B)
    return out
