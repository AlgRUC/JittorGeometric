/*
 * @Description: 
 * @Author: liuyy
 */
#include "var.h"
#include "csctocsr_op.h"
#include <unordered_map>


namespace jittor {
#ifndef JIT
CsctocsrOp::CsctocsrOp(Var* column_offset_,Var* row_indices_,Var* row_offset_,Var* column_indices_,Var* dst_, Var* src_):
column_offset(column_offset_),row_indices(row_indices_),row_offset(row_offset_),column_indices(column_indices_),dst(dst_),src(src_){
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr,row_offset->dtype());
}

void CsctocsrOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "Tint", column_offset->dtype());
    
}
#else // JIT

#ifdef JIT_cpu
void CsctocsrOp::jit_run() {
    //printf("jit_run function called\n"); // 
    auto* __restrict__ c_o = column_offset->ptr<Tint>();
    auto* __restrict__ r_i = row_indices->ptr<Tint>();
    auto* __restrict__ r_o = row_offset->ptr<Tint>();
    auto* __restrict__ c_i = column_indices->ptr<Tint>();

    auto* __restrict__ src_prt = src->ptr<Tint>(); // 不用outpu了，送进来的就是已经排序好，弄好的
    auto* __restrict__ dst_prt = dst->ptr<Tint>();

    Tint src_size=src->shape[0];
    Tint edge_size = row_indices->shape[0];
    Tint dst_size=dst->shape[0];

// step1: 构建 src_id -> dst_id 的哈希映射
constexpr Tint INVALID = -1;
    std::vector<Tint> src_id_to_idx(1 << 20, INVALID); // 若你知道 global_v_num 可设为 global_v_num

    for (Tint i = 0; i < src_size; ++i) {
        src_id_to_idx[src_prt[i]] = i;
    }

    // step1: 统计每个 src 点的出边数，构建 r_o（CSR 的 row offset）
    for (int i = 0; i < dst_size; ++i) {
        for (int j = c_o[i]; j < c_o[i+1]; ++j) {
            Tint src_id = r_i[j];
            Tint row_idx = src_id_to_idx[src_id];
            if (row_idx != INVALID) {
                r_o[row_idx + 1]++;
            }
        }
    }

    // 累加构建 row_offset
    for (int i = 1; i <= src_size; ++i) {
        r_o[i] += r_o[i - 1];
    }

    // 反向写入 CSR 的 column_indices（用一个计数器记录每个 row 的当前写入位置）
    std::vector<Tint> row_ptr(r_o, r_o + src_size);  // 当前每行的写指针（等价于前缀和副本）

    for (int i = 0; i < dst_size; ++i) {
        Tint dst_id = dst_prt[i];
        for (int j = c_o[i]; j < c_o[i+1]; ++j) {
            Tint src_id = r_i[j];
            Tint row_idx = src_id_to_idx[src_id];
            if (row_idx != INVALID) {
                Tint pos = row_ptr[row_idx]++;
                c_i[pos] = dst_id;
            }
        }
    }

    
}
#else // JIT_cpu

__global__ void count_edges_kernel(
    const Tint* src, const Tint* row_indices, 
    Tint* row_offset, Tint src_size, Tint edge_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= src_size) return;

    Tint src_val = src[tid];
    Tint count = 0;
    for (int j = 0; j < edge_size; j++) {
        if (row_indices[j] == src_val) count++;
    }
    row_offset[tid + 1] = count;  // r_o[0] = 0 outside kernel
}


__global__ void prefix_sum_kernel(Tint* row_offset, Tint size) {
    // 只使用一个线程做串行前缀和
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 1; i <= size; ++i) {
        row_offset[i] += row_offset[i - 1];
    }
}

__global__ void build_column_indices_kernel(
    const Tint* src, const Tint* dst,
    const Tint* row_indices, const Tint* column_offset,
    const Tint* row_offset, Tint* column_indices,
    Tint src_size, Tint dst_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= src_size) return;

    Tint src_val = src[tid];
    Tint write_pos = row_offset[tid];

    for (int k = 0; k < dst_size; k++) {
        for (int l = column_offset[k]; l < column_offset[k + 1]; l++) {
            if (row_indices[l] == src_val) {
                column_indices[write_pos++] = dst[k];
            }
        }
    }
}




void CsctocsrOp::jit_run() {
    auto* __restrict__ c_o = column_offset->ptr<Tint>();
    auto* __restrict__ r_i = row_indices->ptr<Tint>();
    auto* __restrict__ r_o = row_offset->ptr<Tint>();
    auto* __restrict__ c_i = column_indices->ptr<Tint>();
    auto* __restrict__ src_ptr = src->ptr<Tint>();
    auto* __restrict__ dst_ptr = dst->ptr<Tint>();
    //printf("jit_gpu function called\n"); // 

    Tint src_size = src->shape[0];
    Tint dst_size = dst->shape[0];
    Tint edge_size = row_indices->shape[0];

    cudaMemset(r_o, 0, sizeof(Tint));

    int threads = 256;
    int blocks_src = (src_size + threads - 1) / threads;

    count_edges_kernel<<<blocks_src, threads>>>(src_ptr, r_i, r_o, src_size, edge_size);


    prefix_sum_kernel<<<1, 1>>>(r_o, src_size);



    build_column_indices_kernel<<<blocks_src, threads>>>(
        src_ptr, dst_ptr, r_i, c_o, r_o, c_i, src_size, dst_size
    );

}






#endif // JIT_cpu


#endif // JIT

} // jittor