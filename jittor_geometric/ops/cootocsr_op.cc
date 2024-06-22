/*
 * @Author: lusz 578752274@qq.com
 * @Date: 2024-06-20 21:41:00
 * @LastEditors: lusz 578752274@qq.com
 * @LastEditTime: 2024-06-20 22:16:54
 * @FilePath: /JittorGNN/jittor_geometric/ops/cootocsr.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "var.h"
#include "cootocsr_op.h"


namespace jittor {
#ifndef JIT
CootocsrOp::CootocsrOp(Var* edge_index_,Var* coo_edge_weight_,Var* column_indices_,Var* row_offset_,Var* csr_edge_weight_,int v_num_,NanoString dtype_) : 
edge_index(edge_index_), coo_edge_weight(coo_edge_weight_),column_indices(column_indices_), row_offset(row_offset_),csr_edge_weight(csr_edge_weight_),dtype(dtype_),v_num(v_num_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

void CootocsrOp::jit_prepare(JK& jk) {
    //std::cout<<myType<<std::endl;
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void CootocsrOp::jit_run() {
    int max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ e_x = edge_index->ptr<int>();
    auto* __restrict__ e_w = coo_edge_weight->ptr<T>();
    auto* __restrict__ e_wr = csr_edge_weight->ptr<T>();
    auto* __restrict__ col_indices = column_indices->ptr<int>();
    auto* __restrict__ row_off = row_offset->ptr<int>();

    int edge_size = edge_index->shape[1];
    // Initialize row_offset
    #pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < edge_size; i++) {
        __sync_fetch_and_add(&row_off[e_x[i] + 1], 1);
    }
    for (int i = 0; i < v_num; i++) {
        row_off[i + 1] += row_off[i];
    }

    int* vertex_index = (int*) calloc(v_num, sizeof(int));
    #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for (int i = 0; i < edge_size; i++)  {
        int src = e_x[i];
        int dst = e_x[i + edge_size];
        int index = __sync_fetch_and_add((int *)&vertex_index[src], 1);
        index += row_off[src];
        col_indices[index] = dst;
        e_wr[index] = e_w[i];
    }
    std::free(vertex_index); // free不在jittor命名空间里
    
    
}
#endif // JIT

} // jittor