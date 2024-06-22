import jittor as jt

header ="""
#pragma once
#include "op.h"
#include <immintrin.h>
namespace jittor {

struct CootocsrOp : Op {
    Var* column_indices;
    Var* row_offset;
    Var* csr_edge_weight; // CSR
    Var* edge_index;
    Var* coo_edge_weight;// COO
    NanoString dtype;
    Var* output; //必须要一个output,暂时没找到解决办法
    int v_num;
    CootocsrOp(Var* edge_index_,Var* coo_edge_weight_,Var* column_indices_,Var* row_offset_,Var*csr_edge_weight_,int v_num_,NanoString dtype_=ns_float32);
    const char* name() const override { return "cootocsr"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "cootocsr_op.h"


namespace jittor {
#ifndef JIT
CootocsrOp::CootocsrOp(Var* edge_index_,Var*coo_edge_weight_,Var* column_indices_,Var* row_offset_,Var*csr_edge_weight_,int v_num_,NanoString dtype_) : 
edge_index(edge_index_), coo_edge_weight(coo_edge_weight_),column_indices(column_indices_), row_offset(row_offset_),csr_edge_weight(csr_edge_weight_),dtype(dtype_),v_num(v_num_){
    flags.set(NodeFlags::_cpu, 1);
    
    output = create_output(nullptr,dtype);
}

void CootocsrOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", dtype);
}

#else // JIT
void CootocsrOp::jit_run() {
    int max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ e_x = edge_index->ptr<int>();
    auto* __restrict__ e_w = edge_weight->ptr<T>();
    auto* __restrict__ e_ws = edge_weight->ptr<T>();
    auto* __restrict__ col_indices = column_indices->ptr<int>();
    auto* __restrict__ row_off = row_offset->ptr<int>();

    int edge_size = edge_index->shape[0];

    // Initialize row_offset
    #pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < edge_size; i += 2) {
        __sync_fetch_and_add(&row_off[e_x[i * 2] + 1], 1);
    }

    for (int i = 0; i < v_num; ++i) {
        row_off[i + 1] += row_off[i];
    }

    int* vertex_index = (int*) calloc(v_num, sizeof(int));
    #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < edge_size; i += 2) {
        int src = e_x[i * 2];
        int dst = e_x[i * 2 + 1];
        int index = __sync_fetch_and_add(&vertex_index[src], 1);
        index += row_off[src];
        col_indices[index] = dst;
        e_ws[index] = e_w[i / 2];
    }
    free(vertex_index);
}
#endif // JIT

} // jittor
"""

def test_coo_to_csr():
    jt.flags.use_cuda = 0
    jt.flags.lazy_execution = 0

    edge_index = jt.array([[0, 2], [1, 0], [2, 1]])
    edge_weight = jt.array([3.0, 4.0, 5.0])
    csr_edge_weight=jt.random((3,))
    column_indices=jt.zeros((3,), dtype='int32')
    row_offset=jt.zeros((3,), dtype='int32')
    v_num = 3

    # Create and compile the custom op
    my_op = jt.compile_custom_op(header, src, "cootocsr",warp=False)
    my_op(edge_index, edge_weight,column_indices,row_offset,csr_edge_weight, v_num,'float32').fetch_sync()


# Run the test
test_coo_to_csr()