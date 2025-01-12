'''
Description: 
Author: lusz
Date: 2024-11-05 16:25:39
'''
import jittor as jt
import os
from jittor_geometric.data import CSR
from jittor_geometric.ops import SpmmCsr
def test_spmm_csr():
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x=jt.array([[3.0, 2.0, 1.0],[3.0, 2.0, 1.0],[3.0, 2.0, 1.0]])
    col_indices=jt.array([0,1,1,2],dtype='int64')
    row_offset=jt.array([0,2,3,4],dtype='int64')
    csr_weight=jt.array([3.0,1.0,4.0,2.0], dtype='float32')
    csr=CSR(column_indices=col_indices,row_offset=row_offset,edge_weight=csr_weight)
    output=SpmmCsr(x,csr)
    print(output)

test_spmm_csr()