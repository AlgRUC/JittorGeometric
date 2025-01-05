'''
Description: 
Author: lusz
Date: 2024-12-28 19:35:50
'''
import jittor as jt
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.ops import SpmmCsr,aggregateWithWeight,cootocsc,cootocsr,SpmmCoo
from jittor_geometric.data import CSC,CSR
def test_spmm_csr():
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x = jt.array([[3.0, 2.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                [1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 1.0, 0.0],
                [0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 0.0],
                [4.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0],
                [7.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0],
                [1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0],
                [0.0, 8.0, 0.0, 0.0, 7.0, 1.0, 0.0, 9.0, 2.0, 0.0],
                [9.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0],
                [0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 5.0, 7.0]], dtype="float32")
    edge_index = jt.array([[0, 0, 1, 1, 2 ,2, 3, 4, 4,5,5,7,8], [1, 2, 2, 3, 3, 4, 5,8,9,5,8,9,9]])
    edge_weight = jt.array([1.0, 2.0, 3.0, 4.0, 5.0,3.0,5.0,1.0,2.0,3.0,2.0,3.0,2.0], dtype="float32")
    csr=cootocsr(edge_index,edge_weight,10)
    csc=cootocsc(edge_index,edge_weight,10)
    # print(csr.column_indices)
    # print(csr.row_offset)
    # print(csr.edge_weight)
    # print(csc.row_indices)
    # print(csc.column_offset)
    # print(csc.edge_weight)
    output_msg = aggregateWithWeight(x,csc,csr)
    print(output_msg)
    output_spmm= SpmmCsr(x,csr)
    print(output_spmm)
    output_coo=SpmmCoo(x,edge_index,edge_weight)
    print(output_coo)
    jt.flags.use_cuda = 0
    output_cpu = aggregateWithWeight(x,csc,csr)
    print(output_cpu)

test_spmm_csr()