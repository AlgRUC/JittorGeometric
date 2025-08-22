/*
 * @Description: 
 * @Author: liuyy
 */
#include "var.h"
#include "sampleprocessing_op.h"
#include "helper_cuda.h"
#include <curand_kernel.h>
#include <stdint.h>

#include <algorithm>
#include <random>


namespace jittor {


#ifndef JIT
SampleprocessingOp::SampleprocessingOp(int fanout_i_,Var* csc_layer_dst_,Var* csc_layer_column_offset_,Var* csc_layer_row_indices_,Var* csc_global_column_offset_,Var* csc_global_row_indices_):
fanout_i(fanout_i_), csc_layer_dst(csc_layer_dst_), csc_layer_column_offset(csc_layer_column_offset_), csc_layer_row_indices(csc_layer_row_indices_), csc_global_column_offset(csc_global_column_offset_), csc_global_row_indices(csc_global_row_indices_){
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr,csc_layer_row_indices->dtype());
        
}
void SampleprocessingOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", csc_layer_dst->dtype());
}
      
    

#else // JIT
#ifdef JIT_cpu
    void SampleprocessingOp::jit_run() {
      //printf("jit_run function called\n"); 
      //int max_threads = std::thread::hardware_concurrency();

      auto* __restrict__ csc_layer_dst_ptr = csc_layer_dst->ptr<T>();
      auto* __restrict__ csc_layer_column_offset_ptr = csc_layer_column_offset->ptr<T>();
      auto* __restrict__ csc_layer_row_indices_ptr = csc_layer_row_indices->ptr<T>();
      auto* __restrict__ csc_global_column_offset_ptr = csc_global_column_offset->ptr<T>();
      auto* __restrict__ csc_global_row_indices_ptr = csc_global_row_indices->ptr<T>();



      //omp_set_num_threads(threads);
      // //#pragma omp parallel for num_threads(threads)
      //#pragma omp parallel for num_threads(max_threads) schedule(guided)
      
            for (int id = 0; id < csc_layer_dst->shape[0]; id++) {
                //printf("正在采样第几个点：%d\n", id); //可以正确输出

                int dst = csc_layer_dst_ptr[id];//发生冲突，错误

                //std::cout << "当前 dst 的值是: " << dst << std::endl;            
              
              
                int edge_nums = csc_global_column_offset_ptr[dst + 1] - csc_global_column_offset_ptr[dst];  
        
              
                int pos = csc_layer_column_offset_ptr[id];
              
                if (fanout_i < edge_nums) {      
                    std::unordered_set<size_t> sampled_idxs;
                    while (sampled_idxs.size() < fanout_i) {//采样第几条边
                    sampled_idxs.insert(rand() % edge_nums);
                    }
                    for (auto& idx : sampled_idxs) {
                      
                        csc_layer_row_indices_ptr[pos++] = csc_global_row_indices_ptr[csc_global_column_offset_ptr[dst] + idx];                        
                      }     
                } else {                            
                  for (size_t i = 0; i < edge_nums; ++i) {                                
                
                csc_layer_row_indices_ptr[pos++] = csc_global_row_indices_ptr[csc_global_column_offset_ptr[dst] + i];
                        
                  }                  
                }                
              }  
              
              

        } 

#else// JIT_cpu

    // 每个 block 负责一个 dst，thread0 做采样并写入
template<int THREADS>
__global__ void sample_gpu_kernel_safe(
    int* sample_indices,            
    const int* sample_offset,       
    const int* destinations,       
    int dst_size,
    const int* gco,                 
    const int* gri,                 
    int fanout,
    int sample_indices_len          // ri length (防越界)
) {
    int id = blockIdx.x;
    if (id >= dst_size) return;

    // 只让 thread0 做随机挑选（fanout 小，成本可接受）
    if (threadIdx.x != 0) return;

    int pos = sample_offset[id];
    int next_pos = sample_offset[id + 1]; //  caller 必须保证长度为 dst_size+1
    if (pos < 0 || pos >= sample_indices_len) return; // 防御性检查
    
    int writable_capacity = next_pos - pos;
    if (writable_capacity <= 0) return;

    int nid = destinations[id];
    
  
    int start = gco[nid];
    int end = gco[nid + 1];
    int deg = end - start;
    if (deg <= 0) return;

    // 初始化 curand
    curandState state;
    // seed：用 clock64() 混合 id 能区分不同 block
    curand_init((unsigned long long)clock64() ^ (unsigned long long)id, 0, 0, &state);

    if (deg <= fanout) {
        
        int write_cnt = min(deg, writable_capacity);
        for (int i = 0; i < write_cnt; ++i) {
            int gid = gri[start + i];
            sample_indices[pos + i] = gid;
        }
        return;
    }

    int local_selected = 0;

    const int MAX_FANOUT = 1024;
    if (fanout > MAX_FANOUT) return; 
    int picked[MAX_FANOUT]; 

    
    while (local_selected < fanout) {
        int r = (int)(curand(&state) % deg); // 0..deg-1
        // 检查是否已选
        bool exist = false;
        for (int k = 0; k < local_selected; ++k) {
            if (picked[k] == r) { exist = true; break; }
        }
        if (!exist) {
            picked[local_selected++] = r;
        }
    }

    // 写回 sample_indices，注意不要写越界
    int write_cnt = min(fanout, writable_capacity);
    for (int i = 0; i < write_cnt; ++i) {
        int idx = picked[i];
        int gid = gri[start + idx];
        sample_indices[pos + i] = gid;
    }
}

void SampleprocessingOp::jit_run() {
    
    auto* dst_ptr = csc_layer_dst->ptr<int>();
    auto* co_ptr  = csc_layer_column_offset->ptr<int>();
    auto* ri_ptr  = csc_layer_row_indices->ptr<int>();
    auto* gco_ptr = csc_global_column_offset->ptr<int>();
    auto* gri_ptr = csc_global_row_indices->ptr<int>();

    int dst_size = (int)csc_layer_dst->shape[0];
    if (dst_size <= 0) return;

    // 传入 ri 长度作为保护
    int ri_len = (int)csc_layer_row_indices->shape[0];

    dim3 blocks(dst_size);
    dim3 threads(32);
    sample_gpu_kernel_safe<32><<<blocks, threads>>>(
        ri_ptr, co_ptr, dst_ptr, dst_size, gco_ptr, gri_ptr, fanout_i, ri_len
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}

#endif //cuda
#endif // JIT

} // jittor





