import jittor as jt
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
addone_op = jt.compile_custom_op(header, src, "addone", warp=False)
class MyFunc(Function):
    def execute(self, x, y):
        self.x = x
        self.y = y
        addone_op(y,x,2.0,1, 'float').fetch_sync()
        return
    def grad(self, grad0, grad1):
        print(1)
        return grad0 * self.y, grad1 * self.x

a = jt.array(3.0)
b = jt.array(4.0)
func = MyFunc.apply
c,d=func(a, b)
da,db = jt.grad(c+d*3, [a, b])
print(da,db)