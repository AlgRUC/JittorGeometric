/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-03 13:50:09
 */

#include  "var.h"
#include  "cub/cub.cuh"
#include  "edgesoftmax_op.h"
typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS_SOFTMAX = 32;
const int CUDA_NUM_BLOCKS_SOFTMAX = 512;

namespace jittor {
#ifndef JIT
EdgesoftmaxOp::EdgesoftmaxOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_) :
outputVar(outputVar_),x(x_),indices(indices_),offset(offset_){
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr,x->dtype());
}


void EdgesoftmaxOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", x->dtype());
     add_jit_define(jk, "Tint", indices->dtype());
}

#else // JIT
    #ifdef JIT_cpu
    void EdgesoftmaxOp::jit_run() {
        auto* __restrict__ out_ptr = outputVar->ptr<T>();
        auto* __restrict__ x_ptr = x->ptr<T>();
        auto* __restrict__ i_ptr = indices->ptr<Tint>();
        auto* __restrict__ o_ptr = offset->ptr<Tint>();
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
    #else //cuda
        template <typename T_v,typename T_l>
        __global__ void edge_softmax_forward_block( T_v* msg_output,  T_v* msg_input,
                        T_v* msg_cached, const T_l *row_indices,const  T_l *column_offset,
                T_l batch_size_, T_l feature_size_){
                int VtxPerBlock=1;
                typedef ::cub::BlockReduce<T_v, CUDA_NUM_THREADS_SOFTMAX> BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                __shared__ T_v bcast_value;
                for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
                    
                    VertexId_CUDA curVtx_trans=blkColStart;
                    VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                    VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];               
                    __syncthreads();
                    T_v thread_data=0.0;
                    int used=0;
                    T_v aggregate = 0.0;
                    for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS_SOFTMAX){

                        int valid_items=rowIdxEnd-rowIdxStart-CUDA_NUM_THREADS_SOFTMAX*used;
                        thread_data=exp(msg_input[eid]);
                        aggregate += thread_data;
                        used+=1;
                    }
                    T_v block_sum = BlockReduce(temp_storage).Sum(aggregate);

                    __syncthreads();
                    if(threadIdx.x==0){
                        bcast_value = block_sum;
                    }
                    __syncthreads();             
                    aggregate=bcast_value;
        
                    for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS_SOFTMAX){   
                        msg_output[eid]=exp(msg_input[eid])/aggregate;
                        msg_cached[eid]=msg_output[eid];
                    }
            }      
        }
       void EdgesoftmaxOp::jit_run() {
            std::cout<<"gpu"<<std::endl;
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ i_ptr = indices->ptr<Tint>();
            auto* __restrict__ o_ptr = offset->ptr<Tint>();
            Tint e_num=indices->shape[1];
            Tint v_num=x->shape[0];
            Tint size=x->shape[1];
            output->set_shape(x->shape);
            auto* __restrict__ y_ptr = output->ptr<T>();;
            edge_softmax_forward_block<float,int><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX>>>(
            out_ptr, x_ptr, y_ptr, i_ptr, o_ptr,
            v_num, size);  // cache y
        }
    #endif //cuda
#endif // JIT

} // jittor