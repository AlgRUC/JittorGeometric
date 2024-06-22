'''
Author: lusz
Date: 2024-06-21 10:21:33
Description: 
'''
import jittor as jt
import os
from jittor import nn
from jittor import Function
current_file_path = os.path.abspath(__file__)
test_path = os.path.dirname(current_file_path)
module_path = os.path.dirname(test_path)
# print(module_path)
src = os.path.join(module_path, "ops/addone_op.cc")
header = os.path.join(module_path, "ops/addone_op.h")

addone_op = jt.compile_custom_ops((src, header))
# Run the test
class MyFunc(Function):
    
    def execute(self,inputVar,weight, feat_size):
        outputVar= jt.zeros(feat_size)
        self.outputVar = outputVar
        self.inputVar = inputVar
        addone_op.addone(outputVar,inputVar,weight,feat_size, 'float').fetch_sync()
        return outputVar

    def grad(self, grad_output):
        print(1)
        # 在反向传播中，输入的梯度就是反向传播的梯度，因为f(x) = x + weight
        # 故 df/dx = 1，因此梯度不变
        return grad_output, None, None
    
jt.flags.lazy_execution = 0
feat_size = 3
weight = 2.0

# Initialize input and output arrays
input_var = jt.array([3.0, 2.0, 1.0])


func = MyFunc()
output_var=func(input_var, weight, feat_size)

# 计算损失并进行反向传播
y = jt.array([1, 1, 2]).float32()
print(output_var)
print(y)
loss = nn.nll_loss(output_var, y)
di = jt.grad(loss, [input_var])

print("Input Variable Gradient:", di)