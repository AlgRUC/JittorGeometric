import jittor as jt
# Create and compile the custom op
# Create and compile the custom op
header = """
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct CootocscOp : Op {
    Var* row_indices;
    Var* column_offset;
    Var* csc_edge_weight; // CSC

    Var* edge_index;
    Var* coo_edge_weight; // COO

    NanoString dtype;
    Var* output; //必须要一个output,暂时没找到解决办法
    int v_num;

    CootocscOp(Var* edge_index_, Var* coo_edge_weight_, Var* row_indices_, Var* column_offset_, Var* csc_edge_weight_, int v_num_, NanoString dtype_ = ns_float32);
    const char* name() const override { return "cootocsc"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
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
"""



def test_coo_to_csc():
    jt.flags.use_cuda = 0
    jt.flags.lazy_execution = 0

    # Constructing a more complex example
    edge_index = jt.array([[0, 0, 1, 1, 2], [1, 2, 2, 3, 3]])
    edge_weight = jt.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(jt.size(edge_index))
    print(jt.size(edge_weight))
    csc_edge_weight = jt.randn(5)
    row_indices = jt.zeros((5,), dtype='int32')
    column_offset = jt.zeros((5,), dtype='int32')
    v_num = 4

    
    
    my_op = jt.compile_custom_op(header, src, "cootocsc", warp=False)
    my_op(edge_index, edge_weight, row_indices, column_offset, csc_edge_weight, v_num, 'float32').fetch_sync()
    
    print("CSC Edge Weight:", csc_edge_weight)
    print("Row Indices:", row_indices)
    print("Column Offset:", column_offset)

test_coo_to_csc()