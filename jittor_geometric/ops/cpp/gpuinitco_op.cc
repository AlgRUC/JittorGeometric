/*
 * @Description: 
 * @Author: liuyy
 */
#include "var.h"
#include "gpuinitco_op.h"
#include "helper_cuda.h"


namespace jittor {


#ifndef JIT
    GpuinitcoOp::GpuinitcoOp(int dst_size_,int fanout_,Var* dst_,Var* csc_layer_column_offset_,Var* csc_global_column_offset_):
    dst_size(dst_size_), fanout(fanout_),dst(dst_),csc_layer_column_offset(csc_layer_column_offset_), csc_global_column_offset(csc_global_column_offset_){
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr,dst->dtype());
        
    }
    void GpuinitcoOp::jit_prepare(JK& jk) {
        add_jit_define(jk, "T", dst->dtype());
    }
      
    
#else // JIT

    #ifdef JIT_cpu
        void GpuinitcoOp::jit_run() {
            auto* __restrict__ dst_ptr = dst->ptr<T>();
            auto* __restrict__ co_ptr = csc_layer_column_offset->ptr<T>();
            auto* __restrict__ gco_ptr = csc_global_column_offset->ptr<T>();
            //printf("JIT_cpu\n"); 
        }
    #else// JIT_cuda


        __global__ void sample_processing_get_co_gpu_kernel(const T *dstt,
            T* local_column_offset,//赋值
            T* global_column_offset,
            T dst_sizee,
            T fanoutt){

            int threadId = blockIdx.x *blockDim.x + threadIdx.x;
            for(int i = threadId; i < dst_sizee; i += blockDim.x * gridDim.x){//不要long啊啊啊
                //printf("kernel running on tid=%d, dst=%d\n", i, dstt[i]);
                T dst_vtx = dstt[i];
                local_column_offset[i + 1] = fminf(global_column_offset[dst_vtx + 1] - global_column_offset[dst_vtx], fanoutt);
                //T m = fminf(global_column_offset[dst_vtx + 1] - global_column_offset[dst_vtx], fanoutt);
                //printf("kernel running on tid=%d, m=%d\n", i, local_column_offset[i + 1]);//可以打印          
            }   
        }


        void GpuinitcoOp::jit_run() {
            auto* __restrict__ dst_ptr = dst->ptr<T>();
            auto* __restrict__ co_ptr = csc_layer_column_offset->ptr<T>();
            auto* __restrict__ gco_ptr = csc_global_column_offset->ptr<T>();



            T blockSize = 256;
            T numBlocks = (dst_size + blockSize - 1) / blockSize;

            //sample_processing_get_co_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADScpu_buffer,0,stream>>>(dst,co_ptr,gco_ptr,dst_size, fanout);
            sample_processing_get_co_gpu_kernel<<<numBlocks, blockSize>>>(dst_ptr,co_ptr,gco_ptr,dst_size,fanout);
            //printf("kernel执行完了\n"); 
            checkCudaErrors(cudaDeviceSynchronize());

 

        }
    #endif//cuda
#endif// JIT
} // jittor





