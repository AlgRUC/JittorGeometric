/*
 * @Author: lusz
 * @Date: 2024-06-21 10:19:23
 * @Description: 
 */

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