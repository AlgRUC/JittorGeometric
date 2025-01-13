'''
Author: lusz
Date: 2024-06-21 10:21:33
Description: 
'''
import jittor as jt
import os
import sys
from jittor import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import ScatterToEdge
jt.flags.use_cuda=1
x=jt.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0],[3.0, 3.0, 3.0],[3.0, 2.0, 1.0]])
y=jt.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
# csc
row_indices=jt.array([0,0,1,2])
col_offset=jt.array([0,1,3,4])
csc_weight=jt.array([1.0,2.0,3.0,4.0])
csc=CSC(row_indices, col_offset, csc_weight)
# csr
col_indices=jt.array([0,1,1,2])
row_offset=jt.array([0,2,3,4])
csr_weight=jt.array([1.0,2.0,3.0,4.0])
csr=CSR(col_indices, row_offset, csr_weight)

# output=ScatterToEdge(x,csc,"src")
output=ScatterToEdge(x,csc,"dst")
# 计算损失并进行反向传播
print(x)
print(output)
# print(y.shape)
loss = nn.BCELoss()
loss_var = loss(output, y)
di = jt.grad(loss_var, [x])

print("Input Variable Gradient:", di)