'''
Author: lusz
Date: 2024-06-21 10:34:31
Description: used for test grad , y=x+weight
'''
import jittor as jt
import os
from jittor import nn
from jittor import Function
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/addone_op.cc")
header = os.path.join(module_path, "cpp/addone_op.h")
addone_op = jt.compile_custom_ops((src, header))

class AddoneFunc(Function):
    
    def execute(self,inputVar,weight, feat_size):
        outputVar= jt.zeros_like(inputVar)
        self.outputVar = outputVar
        self.inputVar = inputVar
        addone_op.addone(outputVar,inputVar,weight,feat_size, 'float').fetch_sync()
        return outputVar

    def grad(self, grad_output):
        return grad_output, None, None
    

def addone(inputVar,weight, feat_size):
    out = AddoneFunc.apply(inputVar,weight, feat_size)
    return out