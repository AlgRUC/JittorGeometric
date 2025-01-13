'''
Author: lusz
Date: 2024-06-20 22:10:53
Description: convert COO to CSR
'''

import jittor as jt
import os
from jittor_geometric.data import CSR
from jittor_geometric.ops import cootocsr

def test_coo_to_csr():
    jt.flags.use_cuda = 0
    jt.flags.lazy_execution = 0

    edge_index = jt.array([[0, 0, 1, 1, 2], [1, 2, 2, 3, 3]])
    edge_weight = jt.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v_num = 4
    csr=cootocsr(edge_index, edge_weight ,v_num)
    
    print("CSR Edge Weight:", csr.edge_weight)
    print("Column Indices:", csr.column_indices)
    print("Row Offset:", csr.row_offset)

test_coo_to_csr()