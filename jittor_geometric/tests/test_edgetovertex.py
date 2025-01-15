'''
Author: lusz
Date: 2024-06-21 10:21:33
Description: 
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import ScatterToVertex

    


jt.flags.use_cuda=1
jt.flags.lazy_execution = 0
x=jt.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0],[3.0, 3.0, 3.0],[3.0, 2.0, 1.0]])
y=jt.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
# csc
row_indices=jt.array([0,0,1,2])
col_offset=jt.array([0,1,3,4])
csc_weight=jt.array([1.0,2.0,3.0,4.0])
csc=CSC(row_indices, col_offset, csc_weight)
output=ScatterToVertex(x,csc,"src")
print(x)
print(output)
# print(y.shape)
loss = nn.BCELoss()
print(output.shape)
print(y.shape)
loss_var = loss(output, y)
di = jt.grad(loss_var, [x])

print("Input Variable Gradient:", di)