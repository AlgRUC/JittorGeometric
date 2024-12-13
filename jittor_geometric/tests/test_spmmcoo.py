'''
Description: 
Author: lusz
Date: 2024-11-11 15:13:39
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.ops import SpmmCoo

def test_spmm_coo():
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x=jt.array([[3.0, 2.0, 1.0],[4.0, 2.0, 2.0],[1.0, 2.0, 3.0]])
    edge_index=jt.array([[0,0,1,2],[1,2,2,1]])
    edge_weight=jt.array([1.0,1.0,1.0,1.0])
    output=SpmmCoo(x,edge_index,edge_weight)
    print(output)

test_spmm_coo()