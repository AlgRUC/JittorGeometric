import time
import jittor as jt
import os
import sys
from jittor import nn
import numpy as np
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSR
from mpi4py import MPI
module_path = os.path.dirname(__file__)

src = os.path.join(module_path, "cpp/distspmm_op.cc")
header = os.path.join(module_path, "cpp/distspmm_op.h")
distspmm_op = jt.compile_custom_ops((src, header))

jt.flags.use_cuda=1

class DistSpmmFunc(Function):
    def execute(self, dist, x, weight, csr_list, local_vnum, trans_A=True, trans_B=False):

        self.dist = dist
        self.csr_list = csr_list
        self.local_vnum = local_vnum
        self.trans_A = trans_A
        self.trans_B = trans_B
        
        self.weight = weight 
        self.input_feature = x 

        z = jt.zeros_like(x)
        feature_bcast = None
        output_part = jt.zeros_like(x)

        for src in range(dist.size):
            if dist.rank == src:
                feature_bcast = x
            feature_bcast = dist.broadcast(x, src)
            
            output_part = jt.zeros_like(x)
            v_num = jt.size(csr_list[src].row_offset, 0) - 1 
            distspmm_op.distspmm(output_part, feature_bcast, csr_list[src].column_indices, csr_list[src].edge_weight, csr_list[src].row_offset, v_num, local_vnum, self.trans_A, trans_B)
            z = z + output_part
        
        self.z = z.fetch_sync()
        self.z=jt.array(self.z)
        self.weight=jt.array(self.weight)
        final_output = jt.matmul(self.z, self.weight)

        return final_output

    def grad(self, grad_output):
        weight_grad_local = jt.matmul(self.z.transpose(), grad_output)
        
        send_buf = weight_grad_local.numpy()
        recv_buf = np.empty_like(send_buf)
        self.dist.comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
        weight_grad = jt.array(recv_buf)
        
        grad_temp_for_feature = jt.matmul(grad_output, self.weight.transpose())
        
        feature_grad = jt.zeros_like(self.input_feature)
        output_part = jt.zeros_like(self.input_feature)

        feature_bcast = None
        for src in range(self.dist.size):
            v_num = jt.size(self.csr_list[src].row_offset, 0) - 1
            if self.dist.rank == src:
                feature_bcast = grad_temp_for_feature
            feature_bcast = self.dist.broadcast(grad_temp_for_feature, src)
            
            output_part = jt.zeros_like(self.input_feature)
            backward_trans_A = not self.trans_A
            distspmm_op.distspmm(output_part, feature_bcast, self.csr_list[src].column_indices, self.csr_list[src].edge_weight, self.csr_list[src].row_offset, v_num, self.local_vnum, backward_trans_A, self.trans_B)

            feature_grad = feature_grad + output_part

        return None, feature_grad, weight_grad, None, None, None, None

def DistSpmm(dist, x, weight, csr_list, local_vnum, trans_A=True, trans_B=False):
    out = DistSpmmFunc.apply(dist, x, weight, csr_list, local_vnum, trans_A, trans_B)
    return out