/*
 * @Description: 
 * @Author: lusz, liyx
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "aggregate_op.h"

namespace jittor {
#ifndef JIT
    AggregateOp::AggregateOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* weight_,bool forward_) :
    outputVar(outputVar_),x(x_),indices(indices_), offset(offset_),weight(weight_),forward(forward_){
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr,x->dtype());
    }

    void AggregateOp::jit_prepare(JK& jk) {
        //std::cout<<myType<<std::endl;
        add_jit_define(jk, "T", x->dtype());
        add_jit_define(jk, "Tint", indices->dtype());
    }

#else // JIT
    #ifdef JIT_cpu
        void AggregateOp::jit_run() {
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ i_ptr = indices->ptr<Tint>();
            auto* __restrict__ o_ptr = offset->ptr<Tint>();
            auto* __restrict__ w_ptr = weight->ptr<T>();
            Tint e_num=indices->shape[1];
            Tint v_num=x->shape[0];
            Tint feature_dim=x->shape[1];
            Tint start;
            Tint end;
            //avx
            #ifdef __AVX__
            const Tint LEN = 8;
            Tint loop = feature_dim / LEN;
            Tint res = feature_dim % LEN;
            for(Tint i=0;i<v_num;i++){
                start=o_ptr[i];
                end=o_ptr[i+1];
                for(Tint j=start;j<end;j++){
                    Tint idx = i_ptr[j] * feature_dim;
                    __m256 w_val = _mm256_set1_ps(w_ptr[j]);
                    for(Tint k=0;k<loop;k++){
                        __m256 x_val = _mm256_loadu_ps(&x_ptr[idx + k*LEN]);
                        __m256 out_val = _mm256_loadu_ps(&out_ptr[i * feature_dim + k*LEN]);
                        _mm256_storeu_ps(&(out_ptr[i * feature_dim+k*LEN]),_mm256_add_ps(_mm256_mul_ps(x_val,w_val),out_val));
                    
                    }
                    for (Tint k = LEN*loop; k < feature_dim; k++) {
                        out_ptr[i * feature_dim + k] += x_ptr[i_ptr[j]*feature_dim+k] * w_ptr[j];
                    }
                    
                }
            }
            #else
            for(Tint i=0;i<v_num;i++){
                start=o_ptr[i];
                end=o_ptr[i+1];
                for(Tint j=start;j<end;j++){
                    for(Tint k=0;k<feature_dim;k++){
                        out_ptr[i*feature_dim+k]=out_ptr[i*feature_dim+k]+x_ptr[i_ptr[j]*feature_dim+k]*w_ptr[j]; 
                    }
                }
            }
            #endif//avx
        }
            
    #else
        __global__ void computeKernel(const Tint* row_indices,const Tint* column_offset,
            const float* old_feature, float* new_feature,const float* weight,
            Tint batch_size_, Tint feature_size_){
            //int large_size=blockDim.x;
            Tint threadId = blockIdx.x * blockDim.x + threadIdx.x;
            for (Tint i = threadId; i < feature_size_ * batch_size_; i += blockDim.x * gridDim.x) {
                if (i >= feature_size_ * batch_size_) return;  // 防止越界
                Tint local_dst = i / feature_size_;
                Tint rank = i % feature_size_;
                // 核心计算部分
                for (Tint i_i = column_offset[local_dst]; i_i < column_offset[local_dst + 1]; i_i++) {
                    Tint local_src = row_indices[i_i];
                    atomicAdd(&new_feature[feature_size_ * local_dst + rank],
                            old_feature[feature_size_ * local_src + rank] * weight[i_i]);
                }
            }
            __syncthreads();
        }

        void AggregateOp::jit_run() {
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ i_ptr = indices->ptr<Tint>();
            auto* __restrict__ o_ptr = offset->ptr<Tint>();
            auto* __restrict__ w_ptr = weight->ptr<T>();
            Tint e_num=indices->shape[1];
            Tint v_num=x->shape[0];
            Tint feature_dim=x->shape[1];
            Tint blockSize = 256;
            // int numBlocks = 128;
            Tint numBlocks = (feature_dim * v_num + blockSize - 1) / blockSize;
            computeKernel<<<numBlocks, blockSize>>>(i_ptr,o_ptr, x_ptr, out_ptr, w_ptr, v_num, feature_dim);
        }

    #endif //cuda
#endif // JIT
} // jittor