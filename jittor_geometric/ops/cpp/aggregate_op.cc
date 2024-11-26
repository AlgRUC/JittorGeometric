/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:03
 */
#include "var.h"
#include "aggregate_op.h"
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }
#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(status); \
    } \
  } while (0)
namespace jittor {
#ifndef JIT
    AggregateOp::AggregateOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* weight_,bool forward_,NanoString dtype_) :
    outputVar(outputVar_),x(x_),indices(indices_), offset(offset_),weight(weight_),forward(forward_),dtype(dtype_){
        flags.set(NodeFlags::_cpu, 1);
        flags.set(NodeFlags::_cuda, 1);
        output = create_output(nullptr,dtype);
    }

    void AggregateOp::jit_prepare(JK& jk) {
        //std::cout<<myType<<std::endl;
        add_jit_define(jk, "T", dtype);
    }

#else // JIT
    #ifdef JIT_cpu
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
            //avx
            #ifdef __AVX__
            // std::cout<<"enabled avx"<<std::endl; 
            const int LEN = 8;
            int loop = feature_dim / LEN;
            int res = feature_dim % LEN;
            for(int i=0;i<v_num;i++){
                start=o_ptr[i];
                end=o_ptr[i+1];
                for(int j=start;j<end;j++){
                    int idx = i_ptr[j] * feature_dim;
                    __m256 w_val = _mm256_set1_ps(w_ptr[j]);
                    for(int k=0;k<loop;k++){
                        __m256 x_val = _mm256_loadu_ps(&x_ptr[idx + k*LEN]);
                        __m256 out_val = _mm256_loadu_ps(&out_ptr[i * feature_dim + k*LEN]);
                        _mm256_storeu_ps(&(out_ptr[i * feature_dim+k*LEN]),_mm256_add_ps(_mm256_mul_ps(x_val,w_val),out_val));
                    
                    }
                    for (int k = LEN*loop; k < feature_dim; k++) {
                        out_ptr[i * feature_dim + k] += x_ptr[i_ptr[j]*feature_dim+k] * w_ptr[j];
                    }
                    
                }
            }
            #else
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
            #endif//avx
        }
            
    #else
        __global__ void computeKernel(const int* row_indices,const int* column_offset,
            const float* old_feature, float* new_feature,const float* weight,
            int batch_size_, int feature_size_){
            //int large_size=blockDim.x;
            int threadId = blockIdx.x * blockDim.x + threadIdx.x;
            for (long i = threadId; i < feature_size_ * batch_size_; i += blockDim.x * gridDim.x) {
                if (i >= feature_size_ * batch_size_) return;  // 防止越界
                int local_dst = i / feature_size_;
                int rank = i % feature_size_;

                // 核心计算部分
                for (int i_i = column_offset[local_dst]; i_i < column_offset[local_dst + 1]; i_i++) {
                    int local_src = row_indices[i_i];
                    atomicAdd(&new_feature[feature_size_ * local_dst + rank],
                            old_feature[feature_size_ * local_src + rank] * weight[i_i]);
                }
            }
        }

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
            const float alpha = 1.0f;
            const float beta = 1.0f;
            int blockSize = 256;
            // int numBlocks = 128;
            int numBlocks = (feature_dim*v_num + blockSize - 1) / blockSize;
            computeKernel<<<numBlocks, blockSize>>>(i_ptr,o_ptr, x_ptr, out_ptr, w_ptr, v_num, feature_dim);
        }
    #endif //cuda
#endif // JIT
} // jittor