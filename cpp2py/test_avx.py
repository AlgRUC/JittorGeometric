import jittor as jt

header ="""
#pragma once
#include "op.h"
#include <immintrin.h>
namespace jittor {

struct CompOp : Op {
    Var* output; //必须要一个output,暂时没找到解决办法
    Var* outputVar;
    Var* inputVar;
    float64 weight;
    int feat_size;
    NanoString myType;
    CompOp(Var* outputVar_,Var* inputVar_,float64 weight_,int feat_size_,NanoString dtype=ns_float32);
    const char* name() const override { return "comp"; }

    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "comp_op.h"


namespace jittor {
#ifndef JIT
CompOp::CompOp(Var* outputVar_,Var* inputVar_,float64 weight_,int feat_size_,NanoString dtype) : 
outputVar(outputVar_), inputVar(inputVar_), weight(weight_), feat_size(feat_size_), myType(dtype){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
    

void CompOp::jit_prepare(JK& jk) {
    //std::cout<<myType<<std::endl;
     add_jit_define(jk, "T", myType);
}

#else // JIT
void CompOp::jit_run() {
    
    auto* __restrict__ x = outputVar->ptr<T>();
    auto *input=inputVar->ptr<T>();
    #ifdef __AVX__  // support AVX 
    std::cout<<"avx enabled"<<std::endl;  
    const int LEN=8;
    int loop=feat_size/LEN;
    int res=feat_size%LEN;
    __m256 w=_mm256_broadcast_ss(reinterpret_cast<float const *>(&weight));
    for(int i=0;i<loop;i++){
        __m256 source= _mm256_loadu_ps(reinterpret_cast<float const *>(&(input[i*LEN])));
        __m256 destination= _mm256_loadu_ps(reinterpret_cast<float const *>(&(x[i*LEN])));
        _mm256_storeu_ps(&(x[i*LEN]),_mm256_add_ps(_mm256_mul_ps(source,w),destination));
    }
    for (int i = LEN*loop; i < feat_size; i++) {
        x[i] += input[i] * weight;
    }
    #else // not support AVX
    for (int i = 0; i < feat_size; i++) {
        x[i] += input[i] * weight;
    }
    #endif
}
#endif // JIT

} // jittor
"""

import numpy as np
# Testing the CustomOp
def test_custom_op():
    feat_size = 3
    weight = 2.0

    # Initialize input and output arrays
    input_var =  jt.Var([3.0,2.0,1.0])
    output_var  = jt.Var([3.0,2.0,1.0])

    # print(input_var)
    # print(output_var)
    comp_op = jt.compile_custom_op(header, src, "comp", warp=False)
    jt.flags.use_cuda = 0
    comp_op(output_var,input_var,weight,feat_size, 'float').fetch_sync()
    print(output_var)

# Run the test
jt.flags.lazy_execution=0
test_custom_op()