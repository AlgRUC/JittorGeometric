/*
 * @Description: 
 * @Author: liuyy
 */
#include "var.h"
#include "getweight_op.h"
#include "helper_cuda.h"



namespace jittor {


#ifndef JIT
GetweightOp::GetweightOp(int vtx_size_, Var* csc_layer_dst_, Var* csc_layer_src_, Var* csc_layer_column_offset_, Var* csc_layer_row_indices_, Var* csc_layer_edge_weight_):
      vtx_size(vtx_size_),
      csc_layer_dst(csc_layer_dst_),
      csc_layer_src(csc_layer_src_),
      csc_layer_column_offset(csc_layer_column_offset_),
      csc_layer_row_indices(csc_layer_row_indices_),
      csc_layer_edge_weight(csc_layer_edge_weight_){
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr,csc_layer_edge_weight->dtype());
        
}
void GetweightOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", csc_layer_dst->dtype());
    add_jit_define(jk, "Tint", csc_layer_dst->dtype());
    add_jit_define(jk, "F", csc_layer_edge_weight->dtype());
}
      
    
#else // JIT

#ifdef JIT_cpu
    void GetweightOp::jit_run() {
      
    auto* __restrict__ dst_ptr = csc_layer_dst->ptr<Tint>();
    auto* __restrict__ src_ptr = csc_layer_src->ptr<Tint>();
    auto* __restrict__ column_offset_ptr = csc_layer_column_offset->ptr<Tint>();
    auto* __restrict__ row_indices_ptr = csc_layer_row_indices->ptr<Tint>();
    auto* __restrict__ edge_weight = csc_layer_edge_weight->ptr<F>();
    Tint max_threads = std::thread::hardware_concurrency();


  
    //更新全图的度，ins和outs
    Tint* ins = new Tint[vtx_size];
    Tint* outs = new Tint[vtx_size];

    // 初始化出入度
    //#pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < vtx_size; ++i) {
        ins[i] = 0;
        outs[i] = 0;
      }


    // 计算入度和出度
    //#pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < csc_layer_dst->shape[0]; ++i) {
        ins[dst_ptr[i]] += column_offset_ptr[i + 1] - column_offset_ptr[i];
        for (int j = column_offset_ptr[i]; j < column_offset_ptr[i + 1]; j++) {
          int src = row_indices_ptr[j];
          outs[src] += 1;
          
        }
      }

    //#pragma omp parallel for num_threads(max_threads) schedule(guided)


      //#pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (Tint i = 0; i < csc_layer_dst->shape[0]; ++i) {
        for (Tint j = column_offset_ptr[i]; j < column_offset_ptr[i + 1]; ++j) {//遍历每条边
            int src = row_indices_ptr[j];
            edge_weight[j] = 1 / (sqrtf((float)outs[src]) * sqrtf((float)ins[dst_ptr[i]]));
            
        }
      }
      delete[] ins;
      delete[] outs;


        


    }
#else// JIT_cpu

    __global__ void re_fresh_degree(T *out_degree,T *in_degree,T vertices){
        int threadId = blockIdx.x *blockDim.x + threadIdx.x;
        for(int i = threadId; i < (int)vertices; i += blockDim.x*gridDim.x){
            out_degree[i] = 0;
            in_degree[i] = 0;
        }
    }

    __global__ void up_date_degree(T *out_degree,T *in_degree,T dstsize,T *destination,T *source,T *column_offset,T *row_indices){
        T threadId = blockIdx.x *blockDim.x + threadIdx.x;

        for(int i = threadId;i < (int)dstsize; i += blockDim.x*gridDim.x){
            int begin_edge = column_offset[i];
            int end_edge = column_offset[i + 1];
            int dst = destination[i];
            in_degree[dst] = end_edge - begin_edge;

            for (int edge = begin_edge; edge < end_edge; edge++) {
                //int src = source[row_indices[edge]];
                int src = row_indices[edge];
                //int src = source[edge];
                atomicAdd(&out_degree[src], 1);
            }
        }
    }


    __global__ void get_weight(float *edge_weight, int *out_degree, int *in_degree, int dstsize,int *destination, int *source, int *column_offset, int *row_indices){
        int threadId = blockIdx.x *blockDim.x + threadIdx.x;

        //for(int warp_id = warpId; warp_id < dstsize; warp_id += warp_num){
            for(int i = threadId;i < (int)dstsize; i += blockDim.x*gridDim.x){
                int begin_edge = column_offset[i];
                int end_edge = column_offset[i + 1];


            //for(int i = start + laneId; i < end; i += WARP_SIZE) {
                for (int edge = begin_edge; edge < end_edge; edge++) {
                //int src = row_indices[edge];
                //int src = edge;
                int src = row_indices[edge];
                edge_weight[edge] = 1 / (sqrtf((float)out_degree[src]) * sqrtf((float)in_degree[destination[i]]));
            }
        }
    }



    void GetweightOp::jit_run() {
        auto* __restrict__ dst_ptr = csc_layer_dst->ptr<T>();
        auto* __restrict__ src_ptr = csc_layer_src->ptr<T>();
        auto* __restrict__ col_offset_ptr = csc_layer_column_offset->ptr<T>();
        auto* __restrict__ row_indices_ptr = csc_layer_row_indices->ptr<T>();
        auto* __restrict__ layer_edge_weight_ptr = csc_layer_edge_weight->ptr<F>();
        int dst_size= csc_layer_dst->shape[0];

        //printf("JIT_gpu\n"); 

        
        //1 给度赋值0
        int* out_degree;
        int* in_degree;
        cudaMalloc(&out_degree, sizeof(int) * vtx_size);
        cudaMalloc(&in_degree, sizeof(int) * vtx_size);

        T blockSize = 256;
        T numBlocks = (vtx_size + blockSize - 1) / blockSize;
        re_fresh_degree<<<numBlocks, blockSize>>>(out_degree,in_degree,vtx_size);
        checkCudaErrors(cudaDeviceSynchronize());
        //printf("给度赋值\n"); 

        //2 计算度
        T blockSize1 = 256;
        T numBlocks1 = (dst_size + blockSize - 1) / blockSize;
        up_date_degree<<<numBlocks, blockSize>>>(out_degree,in_degree,dst_size,dst_ptr,src_ptr,col_offset_ptr,row_indices_ptr);
        checkCudaErrors(cudaDeviceSynchronize());
        //printf("计算度\n"); 

        //3 计算权重

        get_weight<<<numBlocks, blockSize>>>(layer_edge_weight_ptr,out_degree,in_degree,dst_size,dst_ptr,src_ptr,col_offset_ptr,row_indices_ptr);

        //printf("计算权重\n"); 

        cudaFree(out_degree);
        cudaFree(in_degree);

    }
#endif //cuda
#endif// JIT
} // jittor





