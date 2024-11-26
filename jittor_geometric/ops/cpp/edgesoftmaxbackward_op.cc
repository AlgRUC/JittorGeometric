/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-04 16:16:13
 */

#include "var.h"
#include "edgesoftmaxbackward_op.h"


namespace jittor {
#ifndef JIT
EdgesoftmaxbackwardOp::EdgesoftmaxbackwardOp(Var* outputVar_, Var* x_, Var* y_,Var* indices_,Var* offset_,NanoString dtype_) :
outputVar(outputVar_),x(x_),y(y_),indices(indices_),offset(offset_),dtype(dtype_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}


void EdgesoftmaxbackwardOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void EdgesoftmaxbackwardOp::jit_run() {
    auto* __restrict__ out_ptr = outputVar->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    auto* __restrict__ y_ptr = y->ptr<T>();
    auto* __restrict__ i_ptr = indices->ptr<int>();
    auto* __restrict__ o_ptr = offset->ptr<int>();
    int e_num=indices->shape[0];
    int v_num=offset->shape[0]-1;
    int feature_dim=x->shape[1];
    int start;
    int end;
    int n;
    T max_weight;
    T total;
    for(int vtx=0;vtx<v_num;vtx++){
        start=o_ptr[vtx];
        end=o_ptr[vtx+1];
        for (int i = start; i < end; ++i) {
            float dot = 0.0f;
            for (int i = start; i < end; ++i) {
                dot += y_ptr[i] * x_ptr[i];
            }
            for (int i = start; i < end; ++i) {
                out_ptr[i] = (y_ptr[i] * x_ptr[i]) - y_ptr[i] * dot;
            }
        }
    }
}
#endif // JIT

} // jittor