/*
 * @Description:
 * @Author: lusz
 * @Date: 2024-06-21 20:20:17
 */
#include "var.h"
#include "cootocsc_op.h"

namespace jittor {
#ifndef JIT
CootocscOp::CootocscOp(Var* edge_index_, Var* coo_edge_weight_, Var* row_indices_, Var* column_offset_, Var* csc_edge_weight_, int v_num_) :
edge_index(edge_index_), coo_edge_weight(coo_edge_weight_), row_indices(row_indices_), column_offset(column_offset_), csc_edge_weight(csc_edge_weight_),v_num(v_num_) {
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr, coo_edge_weight->dtype());
}

void CootocscOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", coo_edge_weight->dtype());
    add_jit_define(jk, "Tint", edge_index->dtype());
}

#else // JIT
void CootocscOp::jit_run() {
    Tint max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ e_x = edge_index->ptr<Tint>();
    auto* __restrict__ e_w = coo_edge_weight->ptr<T>();
    auto* __restrict__ e_wr = csc_edge_weight->ptr<T>();
    auto* __restrict__ r_i = row_indices->ptr<Tint>();
    auto* __restrict__ col_off = column_offset->ptr<Tint>();

    Tint edge_size = edge_index->shape[1];
    // #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (Tint i = 0; i < edge_size; i++) {
        __sync_fetch_and_add(&col_off[e_x[i + edge_size] + 1], 1);
    }
    
    for (Tint i = 0; i < v_num; ++i) {
        col_off[i + 1] += col_off[i];
    }

    Tint* vertex_index = (Tint*) calloc(v_num, sizeof(Tint));
    // #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (Tint i = 0; i < edge_size; i++) {
        Tint src = e_x[i];
        Tint dst = e_x[i + edge_size];
        Tint index = __sync_fetch_and_add((Tint *)&vertex_index[dst], 1);
        __sync_fetch_and_add((Tint *)&index, col_off[dst]);
        // index += col_off[dst];
        r_i[index] = src;
        e_wr[index] = e_w[i];
    }
    std::free(vertex_index);
}
#endif // JIT

} // jittor