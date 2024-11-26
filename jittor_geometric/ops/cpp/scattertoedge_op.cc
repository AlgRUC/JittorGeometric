/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "scattertoedge_op.h"


namespace jittor {
#ifndef JIT
ScattertoedgeOp::ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* edge_weight_,bool with_weight_,int flag_,NanoString dtype_) :
outputVar(outputVar_),x(x_),indices(indices_),offset(offset_),edge_weight(edge_weight_),with_weight(with_weight_),dtype(dtype_),flag(flag_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

ScattertoedgeOp::ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_, bool with_weight_,int flag_,NanoString dtype_) :
    outputVar(outputVar_), x(x_), indices(indices_),offset(offset_), edge_weight(nullptr), with_weight(with_weight_), dtype(dtype_),flag(flag_) {
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr, dtype);
}

void ScattertoedgeOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void ScattertoedgeOp::jit_run() {
    auto* __restrict__ out_ptr = outputVar->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    auto* __restrict__ i_ptr =indices->ptr<int>();
    auto* __restrict__ o_ptr =offset->ptr<int>();
    int e_num=indices->shape[0];
    int feature_dim=x->shape[1];
    int v_num=offset->shape[0]-1;
    int node;
    if(edge_weight!=nullptr){
        auto* __restrict__ e_w=edge_weight->ptr<int>();
        if(flag==0){
        for(int vtx=0;vtx<v_num;vtx++){
            for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                node=i_ptr[i];
                for(int j=0;j<feature_dim;j++){
                    out_ptr[i*feature_dim+j]=out_ptr[i*feature_dim+j]+x_ptr[node*feature_dim+j]/e_w[i];
                }
            }
        }
    }
        if(flag==1){
            // dst
            for(int vtx=0;vtx<v_num;vtx++){
                for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                    for(int j=0;j<feature_dim;j++){
                        out_ptr[i*feature_dim+j]=out_ptr[i*feature_dim+j]+x_ptr[vtx*feature_dim+j]/e_w[i];
                    }
                }
            }
        }
    }
    else{
        if(flag==0){
            for(int vtx=0;vtx<v_num;vtx++){
                for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                    node=i_ptr[i];
                    for(int j=0;j<feature_dim;j++){
                        out_ptr[i*feature_dim+j]=out_ptr[i*feature_dim+j]+x_ptr[node*feature_dim+j];
                    }
                }
             }
        }
        if(flag==1){
            // dst
            for(int vtx=0;vtx<v_num;vtx++){
                for(int i=o_ptr[vtx];i<o_ptr[vtx+1];i++){
                    for(int j=0;j<feature_dim;j++){
                        out_ptr[i*feature_dim+j]=out_ptr[i*feature_dim+j]+x_ptr[vtx*feature_dim+j];
                    }
                }
            }
        }
    }
    
    
    
    
}
#endif // JIT

} // jittor