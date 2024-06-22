'''
Author: lusz
Date: 2024-06-20 16:45:22
LastEditTime: 2024-06-21 10:02:14
Description: used for test, Var add weight
FilePath: /JittorGNN/cpp2py/test_addone.py
'''

import jittor as jt
from jittor import nn
from jittor import Function
header ="""
#pragma once
#include "op.h"
#include <immintrin.h>
namespace jittor {

struct AddoneOp : Op {
    Var* output; //必须要一个output,暂时没找到解决办法
    Var* outputVar;
    Var* inputVar;
    float64 weight;
    int feat_size;
    NanoString myType;
    AddoneOp(Var* outputVar_,Var* inputVar_,float64 weight_,int feat_size_,NanoString dtype=ns_float32);
    const char* name() const override { return "addone"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "addone_op.h"


namespace jittor {
#ifndef JIT
AddoneOp::AddoneOp(Var* outputVar_,Var* inputVar_,float64 weight_,int feat_size_,NanoString dtype) : 
outputVar(outputVar_), inputVar(inputVar_), weight(weight_), feat_size(feat_size_), myType(dtype){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

void AddoneOp::jit_prepare(JK& jk) {
    //std::cout<<myType<<std::endl;
     add_jit_define(jk, "T", myType);
}

#else // JIT
void AddoneOp::jit_run() {
    
    auto* __restrict__ x = outputVar->ptr<T>();
    auto *input=inputVar->ptr<T>();
    for (int i = 0; i < feat_size; i++) {
        x[i] = input[i] +weight;
    }
}
#endif // JIT

} // jittor
"""

import numpy as np

addone_op = jt.compile_custom_op(header, src, "addone", warp=False)
# Run the test
class MyFunc(Function):
    
    def execute(self,intputVar,weight, feat_size):
        outputVar= jt.zeros(feat_size)
        self.outputVar = outputVar
        self.intputVar = intputVar
        addone_op(outputVar,input_var,weight,feat_size, 'float').fetch_sync()
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