/*
 * @Author: lusz
 * @Date: 2025-04-26 22:34:12
 * @Description: 
 */
#include "var.h"
#include "indexselectbwd_op.h"

namespace jittor {
#ifndef JIT
    IndexselectbwdOp::IndexselectbwdOp(Var* outputVar_, Var* x_,int dim_,Var* index_) :
    outputVar(outputVar_),x(x_),dim(dim_),index(index_){
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr,x->dtype());
    }

    void IndexselectbwdOp::jit_prepare(JK& jk) {
        add_jit_define(jk, "T", x->dtype());
        add_jit_define(jk, "Tint", index->dtype());
    }

#else // JIT
    #ifdef JIT_cpu
        void IndexselectbwdOp::jit_run() {
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ i_ptr = index->ptr<Tint>();
            Tint i_num=index->shape[0];
            Tint feature_dim=x->shape[1];
            for(int i=0;i<i_num;i++){
                for(int j=0;j<feature_dim;j++){
                    out_ptr[i_ptr[i]*feature_dim+j]=x_ptr[i*feature_dim+j];
                }
            }
        }
            
    #else
    __global__ void computeKernel(const Tint* __restrict__ i_ptr, 
        const T* __restrict__ x_ptr, T* __restrict__ out_ptr,Tint i_num, Tint feature_dim) {
        Tint tid = blockIdx.x * blockDim.x + threadIdx.x;
        Tint total = i_num * feature_dim;
        for (Tint idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
            Tint i = idx / feature_dim;
            Tint j = idx % feature_dim;
            out_ptr[i_ptr[i] * feature_dim + j] = x_ptr[idx];
        }
        __syncthreads();
    }

    void IndexselectbwdOp::jit_run() {
        auto* __restrict__ out_ptr = outputVar->ptr<T>();
        auto* __restrict__ x_ptr   = x->ptr<T>();
        auto* __restrict__ i_ptr   = index->ptr<Tint>();
        Tint i_num = index->shape[0];
        Tint feature_dim = x->shape[1];
        Tint blockSize = 256;
        Tint numBlocks = (i_num * feature_dim + blockSize - 1) / blockSize;
        computeKernel<<<numBlocks, blockSize>>>(i_ptr, x_ptr, out_ptr, i_num, feature_dim);
    }

    #endif //cuda
#endif // JIT
} // jittor