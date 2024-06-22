/*
 * @Description:
 * @Author: lusz
 * @Date: 2024-06-21 20:20:17
 */
#include "var.h"
#include "cootocsc_op.h"

namespace jittor {
#ifndef JIT
CootocscOp::CootocscOp(Var* edge_index_, Var* coo_edge_weight_, Var* row_indices_, Var* column_offset_, Var* csc_edge_weight_, int v_num_, NanoString dtype_) :
edge_index(edge_index_), coo_edge_weight(coo_edge_weight_), row_indices(row_indices_), column_offset(column_offset_), csc_edge_weight(csc_edge_weight_), dtype(dtype_), v_num(v_num_) {
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr, dtype);
}

void CootocscOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", dtype);
}

#else // JIT
void CootocscOp::jit_run() {
    int max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ e_x = edge_index->ptr<int>();
    auto* __restrict__ e_w = coo_edge_weight->ptr<T>();
    auto* __restrict__ e_wr = csc_edge_weight->ptr<T>();
    auto* __restrict__ r_i = row_indices->ptr<int>();
    auto* __restrict__ col_off = column_offset->ptr<int>();

    int edge_size = edge_index->shape[1];
    // Initialize column_offset
    #pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < edge_size; i++) {
        __sync_fetch_and_add(&col_off[e_x[i + edge_size] + 1], 1);
    }
    for (int i = 0; i < v_num; ++i) {
        col_off[i + 1] += col_off[i];
    }

    int* vertex_index = (int*) calloc(v_num, sizeof(int));
    #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < edge_size; i++) {
        int src = e_x[i];
        int dst = e_x[i + edge_size];
        int index = __sync_fetch_and_add((int *)&vertex_index[dst], 1);
        index += col_off[dst];
        r_i[index] = src;
        e_wr[index] = e_w[i];
    }
    std::free(vertex_index);
}
#endif // JIT

} // jittor