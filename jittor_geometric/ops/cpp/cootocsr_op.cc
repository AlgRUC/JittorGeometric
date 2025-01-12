/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 20:20:26
 */
#include "var.h"
#include "cootocsr_op.h"


namespace jittor {
#ifndef JIT
CootocsrOp::CootocsrOp(Var* edge_index_,Var* coo_edge_weight_,Var* column_indices_,Var* row_offset_,Var* csr_edge_weight_,int v_num_) : 
edge_index(edge_index_), coo_edge_weight(coo_edge_weight_),column_indices(column_indices_), row_offset(row_offset_),csr_edge_weight(csr_edge_weight_),v_num(v_num_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,coo_edge_weight->dtype());
}

void CootocsrOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", coo_edge_weight->dtype());
    add_jit_define(jk, "Tint", edge_index->dtype());
}

#else // JIT
void CootocsrOp::jit_run() {
    Tint max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ e_x = edge_index->ptr<Tint>();
    auto* __restrict__ e_w = coo_edge_weight->ptr<T>();
    auto* __restrict__ e_wr = csr_edge_weight->ptr<T>();
    auto* __restrict__ col_indices = column_indices->ptr<Tint>();
    auto* __restrict__ row_off = row_offset->ptr<Tint>();

    Tint edge_size = edge_index->shape[1];
    // Initialize row_offset
    // #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < edge_size; i++) {
        __sync_fetch_and_add(&row_off[e_x[i] + 1], 1);
    }
    
    for (int i = 0; i < v_num; i++) {
        row_off[i + 1] += row_off[i];
    }

    Tint* vertex_index = (Tint*) calloc(v_num, sizeof(Tint));
    // #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < edge_size; i++)  {
        Tint src = e_x[i];
        Tint dst = e_x[i + edge_size];
        Tint index = __sync_fetch_and_add((Tint *)&vertex_index[src], 1);
        __sync_fetch_and_add((Tint *)&index, row_off[src]);
        // index += row_off[src];
        col_indices[index] = dst;
        e_wr[index] = e_w[i];
    }
    std::free(vertex_index); // free不在jittor命名空间里
    
    
}
#endif // JIT

} // jittor