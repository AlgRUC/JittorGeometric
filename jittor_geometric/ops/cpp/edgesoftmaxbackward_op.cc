/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-04 16:16:13
 */

#include "var.h"
#include "edgesoftmaxbackward_op.h"
typedef uint32_t VertexId_CUDA;

namespace jittor {
#ifndef JIT
EdgesoftmaxbackwardOp::EdgesoftmaxbackwardOp(Var* outputVar_, Var* x_, Var* y_,Var* indices_,Var* offset_) :
outputVar(outputVar_),x(x_),y(y_),indices(indices_),offset(offset_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,x->dtype());
}


void EdgesoftmaxbackwardOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", x->dtype());
     add_jit_define(jk, "Tint", indices->dtype());
}

#else // JIT
    #ifdef JIT_cpu
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
    #else //cuda
    template <typename T_v,typename T_l>
    __global__ void edge_softmax_backward_block( T_v* msg_input_grad, T_v* msg_output_grad,
                    T_v* msg_cached, const T_l *row_indices,const  T_l *column_offset,
            T_l batch_size_, T_l feature_size_){
            int VtxPerBlock=1;
            typedef cub::BlockReduce<T_v, CUDA_NUM_THREADS_SOFTMAX> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ T_v bcast_value;
            
            for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
                VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x/CUDA_NUM_THREADS_SOFTMAX;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];               
                __syncthreads();
                T_v thread_data=0.0;
                int rest=0;
                T_v aggregate=0;
                for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS_SOFTMAX){       
                    int valid_items=rowIdxEnd-rowIdxStart-CUDA_NUM_THREADS_SOFTMAX*rest;
                    thread_data=msg_output_grad[eid]*msg_cached[eid];
                    aggregate += thread_data;
                    rest+=1;
                }
                T_v block_sum = BlockReduce(temp_storage).Sum(aggregate);
                __syncthreads();
                if(threadIdx.x==0){
                    bcast_value=block_sum;
                } 
                __syncthreads();
                T_v aggregate_1=bcast_value;
                for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS_SOFTMAX){       
                    msg_input_grad[eid]=msg_output_grad[eid]*msg_cached[eid]-aggregate_1*msg_cached[eid];
                }
        }      
    }

    void EdgesoftmaxbackwardOp::jit_run() {
            std::cout<<"gpu"<<std::endl;
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ y_ptr = y->ptr<T>();
            auto* __restrict__ i_ptr = indices->ptr<int>();
            auto* __restrict__ o_ptr = offset->ptr<int>()
            Tint e_num=indices->shape[1];
            Tint v_num=x->shape[0];
            Tint size=x->shape[1];
            const int CUDA_NUM_THREADS_SOFTMAX = 32;
            const int CUDA_NUM_BLOCKS_SOFTMAX = 512;
            edge_softmax_backward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX>>>(
            x_ptr, out_ptr, y_ptr, i_ptr, o_ptr,
            v_num, size); 
        }
    #endif// cuda
#endif // JIT

} // jittor