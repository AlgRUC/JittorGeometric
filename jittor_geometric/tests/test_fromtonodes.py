'''
Description: 
Author: lusz
Date: 2025-01-11 13:51:38
'''
import jittor as jt
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import from_nodes,to_nodes

col_offset=jt.array([0,2,4,5,6,7])
row_indices=jt.array([0,3,1,4,0,4,2])
csc_weight=None
csc=CSC(row_indices, col_offset, csc_weight)

row_offset=jt.array([0,2,3,4,5,7])
col_indices=jt.array([0,2,1,4,0,1,3])
csr_weight=None
csr=CSR(col_indices, row_offset, csr_weight)

nodes = jt.array([1, 2, 4])
result1 = from_nodes(csc=csc, nodes=nodes)
result2 = to_nodes(csr=csr ,nodes=nodes)
print(result1) # 0 1 2 4
print(result2) # 1 3 4

