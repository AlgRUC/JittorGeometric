/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "edgetovertex_op.h"


namespace jittor {
#ifndef JIT

EdgetovertexOp::EdgetovertexOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_, int flag_,NanoString dtype_) :
    outputVar(outputVar_), x(x_), indices(indices_),offset(offset_),dtype(dtype_),flag(flag_) {
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr, dtype);
}

void EdgetovertexOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void EdgetovertexOp::jit_run() {
    auto* __restrict__ out_ptr = outputVar->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    auto* __restrict__ i_ptr=indices->ptr<int>();
    auto* __restrict__ o_ptr=offset->ptr<int>();
    int e_num=indices->shape[0];
    int v_num=offset->shape[0]-1;
    int feature_dim=x->shape[1];
    int node;
    if(flag==0){
        for(int vtx=0;vtx<v_num;vtx++){
            for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                node=i_ptr[i];
                for(int j=0;j<feature_dim;j++){
                    out_ptr[node*feature_dim+j]=out_ptr[node*feature_dim+j]+x_ptr[i*feature_dim+j];
                }
            }
        }
    }
    if(flag==1){
        // dst
        for(int vtx=0;vtx<v_num;vtx++){
            for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                for(int j=0;j<feature_dim;j++){
                    out_ptr[vtx*feature_dim+j]=out_ptr[vtx*feature_dim+j]+x_ptr[i*feature_dim+j];;
                }
            }
        }
    }   
}
#endif // JIT

} // jittor