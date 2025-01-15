/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "scattertoedge_op.h"
#ifdef HAS_CUDA
#include "helper_cuda.h"
#endif

namespace jittor {
#ifndef JIT
    ScattertoedgeOp::ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* edge_weight_,bool with_weight_,int flag_) :
    outputVar(outputVar_),x(x_),indices(indices_),offset(offset_),edge_weight(edge_weight_),with_weight(with_weight_),flag(flag_){
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr,x->dtype());
    }

    ScattertoedgeOp::ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_, bool with_weight_,int flag_) :
        outputVar(outputVar_), x(x_), indices(indices_),offset(offset_), edge_weight(nullptr), with_weight(with_weight_), flag(flag_) {
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr, x->dtype());
    }

    void ScattertoedgeOp::jit_prepare(JK& jk) {
        add_jit_define(jk, "T", x->dtype());
        add_jit_define(jk, "Tint", indices->dtype());
    }

#else // JIT
    #ifdef JIT_cpu
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
    #else // cuda
            template <typename T_v,typename T_l>
            __global__ void scatter_dst_to_msg( T_v* message, T_v* dst_feature,
                            const T_l *row_indices,const  T_l *column_offset,
                    T_l batch_size_, T_l feature_size_){
                int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
                for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
                        T_l local_dst=i/feature_size_;
                        T_l rank=i%feature_size_;
                        for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                            message[feature_size_*i_i+rank]=
                                    dst_feature[feature_size_*local_dst+rank];
                        }	
                }
            }

            template <typename T_v,typename T_l>
            __global__ void scatter_src_to_msg( T_v* message, T_v* src_feature,
                            const T_l *row_indices,const  T_l *column_offset,
                    T_l batch_size_, T_l feature_size_){
                int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
                for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
                        T_l local_dst=i/feature_size_;
                        T_l rank=i%feature_size_;
                        for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                            T_l local_src=row_indices[i_i];
                            message[feature_size_*i_i+rank]=
                                    src_feature[feature_size_*local_src+rank];
                        }	
                }
            }
            void ScattertoedgeOp::jit_run() {
                // std::cout<<"v2e gpu"<<std::endl; 
                auto* __restrict__ out_ptr = outputVar->ptr<T>();
                auto* __restrict__ x_ptr = x->ptr<T>();
                auto* __restrict__ i_ptr = indices->ptr<Tint>();
                auto* __restrict__ o_ptr = offset->ptr<Tint>();
                Tint v_num=x->shape[0];
                Tint feature_dim=x->shape[1];
                Tint blockSize = 256;
                // int numBlocks = 128;
                if(flag==1){
                    Tint numBlocks = (feature_dim * v_num + blockSize - 1) / blockSize;
                scatter_dst_to_msg<T,Tint><<<numBlocks,blockSize>>>(out_ptr, x_ptr, i_ptr, o_ptr,v_num, feature_dim);
                }
                else{
                    Tint numBlocks = (feature_dim * v_num + blockSize - 1) / blockSize;
                scatter_src_to_msg<T,Tint><<<numBlocks,blockSize>>>(out_ptr, x_ptr, i_ptr, o_ptr,v_num, feature_dim);
                }
            
                // checkCudaErrors(cudaDeviceSynchronize());
            }
            

    #endif //cuda


#endif // JIT

} // jittor