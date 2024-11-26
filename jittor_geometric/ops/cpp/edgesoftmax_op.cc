/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-03 13:50:09
 */

#include "var.h"
#include "edgesoftmax_op.h"


namespace jittor {
#ifndef JIT
EdgesoftmaxOp::EdgesoftmaxOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,NanoString dtype_) :
outputVar(outputVar_),x(x_),indices(indices_),offset(offset_),dtype(dtype_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}


void EdgesoftmaxOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void EdgesoftmaxOp::jit_run() {
    auto* __restrict__ out_ptr = outputVar->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    auto* __restrict__ i_ptr = indices->ptr<int>();
    auto* __restrict__ o_ptr = offset->ptr<int>();
    int e_num=indices->shape[0];
    int v_num=offset->shape[0]-1;
    int feature_dim=x->shape[1];
    int start;
    int end;
    T max_weight;
    T total;
    for(int vtx=0;vtx<v_num;vtx++){
        start=o_ptr[vtx];
        end=o_ptr[vtx+1];
        total=0;
        max_weight=x_ptr[start];
        for(int i=start;i<end;i++){
            if(max_weight<x_ptr[i]){
                max_weight=x_ptr[i];
            }
        }
        for(int i=start;i<end;i++){
            out_ptr[i]=x_ptr[i]-max_weight;
            total+=exp(out_ptr[i]);
        }
        for(int i=start;i<end;i++){
            out_ptr[i]=exp(out_ptr[i])/total;
        }
    }
}
#endif // JIT

} // jittor