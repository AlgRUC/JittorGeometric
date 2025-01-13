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
from jittor_geometric.data import CSC, CSR
current_file_path = os.path.abspath(__file__)
test_path = os.path.dirname(current_file_path)
module_path = os.path.dirname(test_path)
# print(module_path)
src = os.path.join(module_path, "ops/cpp/aggregate_op.cc")
header = os.path.join(module_path, "ops/cpp/aggregate_op.h")

aggregate_op = jt.compile_custom_ops((src, header))
# Run the test
class MyFunc(Function):
    def execute(self,x,csc,csr):
        self.csc=csc
        self.csr=csr
        edge_weight=csc.edge_weight
        indices=csc.row_indices
        offset=csc.column_offset
        dtype=edge_weight.dtype
        output=x
        aggregate_op.aggregate(output,x,indices,offset,edge_weight,True).fetch_sync()
        return output

    def grad(self, grad_output):
        edge_weight=self.csr.edge_weight
        indices=self.csr.column_indices
        offset=self.csr.row_offset
        dtype=edge_weight.dtype
        output_grad=grad_output
        aggregate_op.aggregate(output_grad,grad_output,indices,offset,edge_weight,False).fetch_sync()
        return output_grad,None,None
    
jt.flags.lazy_execution = 0
x=jt.array([[3.0, 2.0, 1.0],[3.0, 2.0, 1.0],[3.0, 2.0, 1.0],[3.0, 2.0, 1.0]])
y=jt.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
# csc
row_indices=jt.array([0,0,1,2])
col_offset=jt.array([0,1,3,4])
csc_weight=jt.array([1.0,2.0,3,0,4.0])
csc=CSC(row_indices, col_offset, csc_weight)
# csr
col_indices=jt.array([0,1,1,2])
row_offset=jt.array([0,2,3,4])
csr_weight=jt.array([3.0,1.0,4,0,2.0])
csr=CSR(col_indices, row_offset, csr_weight)

func = MyFunc()
print("x")
abs_x=x.abs().sum()
print(abs_x)
output=func(x,csc,csr)
print("out")
abs_out=output.abs().sum()
print(abs_out)

# 计算损失并进行反向传播
print(output.shape)
print(y.shape)
loss = nn.BCELoss()
loss_var = loss(output, y)
di = jt.grad(loss_var, [x])

print("Input Variable Gradient:", di)