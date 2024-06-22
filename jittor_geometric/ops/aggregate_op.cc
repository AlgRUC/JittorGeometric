/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "aggregate_op.h"


namespace jittor {
#ifndef JIT
AggregateOp::AggregateOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* weight_,bool forward_,NanoString dtype_) :
outputVar(outputVar_),x(x_),indices(indices_), offset(offset_),weight(weight_),forward(forward_),dtype(dtype_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

void AggregateOp::jit_prepare(JK& jk) {
    //std::cout<<myType<<std::endl;
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void AggregateOp::jit_run() {
    auto* __restrict__ out_ptr = outputVar->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    auto* __restrict__ i_ptr = indices->ptr<int>();
    auto* __restrict__ o_ptr = offset->ptr<int>();
    auto* __restrict__ w_ptr = weight->ptr<T>();
    int e_num=indices->shape[1];
    int v_num=x->shape[0];
    int feature_dim=x->shape[1];
    int start;
    int end;
    // 待加速 AVX 
    for(int i=0;i<v_num;i++){
        start=o_ptr[i];
        end=o_ptr[i+1];
        for(int j=start;j<end;j++){
            for(int k=0;k<feature_dim;k++){
                out_ptr[i*feature_dim+k]=out_ptr[i*feature_dim+k]+x_ptr[i_ptr[j]*feature_dim+k]*w_ptr[j];
            }
            
        }
    }
}
#endif // JIT

} // jittor