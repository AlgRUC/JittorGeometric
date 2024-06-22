import jittor as jt

header ="""
#pragma once
#include "op.h"

namespace jittor {

struct TestOp : Op {
    Var* output;
    Var* Input1Var;
    Var* Input2Var;
    NanoString myType;
    TestOp(Var* Input1Var_,Var* Input2Var_,float64 weight,NanoVector shape, NanoString dtype=ns_float32);
    
    const char* name() const override { return "test"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "test_op.h"

namespace jittor {
#ifndef JIT
TestOp::TestOp(Var* Input1Var_,Var* Input2Var_,float64 weight,NanoVector shape, NanoString dtype) {
    myType=dtype;
    Input1Var=Input1Var_;
    Input2Var=Input2Var_;
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(shape, dtype);
}

void TestOp::jit_prepare(JK& jk) {
    
    std::cout<<output->dtype()<<std::endl;
    std::cout<<Input1Var->dtype()<<std::endl;
    std::cout<<Input2Var->dtype()<<std::endl;
    add_jit_define(jk, "T", myType); //这种写法没毛病
}

#else // JIT
void TestOp::jit_run() {
    index_t num = output->num;
    // std::cout<<output->num<<std::endl; 60
    auto* __restrict__ y = output->ptr<T>();
    auto* __restrict__ x1 = Input1Var->ptr<T>();
    auto* __restrict__ x2 = Input2Var->ptr<T>();
    for (index_t i=0; i<num; i++){
        y[i] = (T)x1[i]+(T)x2[i];
    }
        
}
#endif // JIT

} // jittor
"""

my_op = jt.compile_custom_op(header, src, "test", warp=False)
input1_var  = jt.Var([3.0,2.0,1.0])
input2_var  = jt.Var([3.0,2.0,1.0])
weight=2.0
# run cpu version
jt.flags.use_cuda = 0
a = my_op(input1_var,input2_var,weight,[3], 'float').fetch_sync()
# assert (a.flatten() == range(3*4*5)).all()
print(a)