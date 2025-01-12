/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-23 16:06:20
 */
#include "var.h"
#include "toundirected_op.h"


namespace jittor {
#ifndef JIT
ToundirectedOp::ToundirectedOp(Var* edge_index_,Var* edge_attr_,int num_edges_,int num_nodes_,Var* new_edge_index_,Var* new_edge_attr_,NanoString dtype_):
edge_index(edge_index_),edge_attr(edge_attr_),num_edges(num_edges_),num_nodes(num_nodes_),new_edge_index(new_edge_index_),new_edge_attr(new_edge_attr_),dtype(dtype_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

void ToundirectedOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", dtype);
    
}
#else // JIT
struct Edge {
    int row;
    int col;
    T data;
};
bool edge_less(const Edge& e1, const Edge& e2) {
    if (e1.row != e2.row)
        return e1.row < e2.row;
    return e1.col < e2.col;
}
void ToundirectedOp::jit_run() {
    auto* __restrict__ e_x = edge_index->ptr<int>();
    auto* __restrict__ e_a = edge_attr->ptr<T>();
    std::vector<Edge> edges;
    for (int i = 0; i < num_edges; ++i) {
        edges.push_back({ e_x[i], e_x[i+num_edges], e_a[i] });
        edges.push_back({ e_x[i+num_edges], e_x[i], e_a[i] });
    }
    std::sort(edges.begin(), edges.end(), edge_less);
    edges.erase(std::unique(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) {
        return e1.row == e2.row && e1.col == e2.col;
    }), edges.end());
    NanoVector index_shape;
    NanoVector attr_shape;
    int length=edges.size();
    index_shape.push_back(2);
    index_shape.push_back(length);
    attr_shape.push_back(1);
    attr_shape.push_back(length);
    new_edge_index->set_shape(index_shape);
    new_edge_attr->set_shape(attr_shape);
    auto* __restrict__ n_e_x = new_edge_index->ptr<int>();
    auto* __restrict__ n_e_a = new_edge_attr->ptr<T>();
    for(int i=0 ; i<length ; i++){
        n_e_x[i]=edges[i].row;
        n_e_x[i+length]=edges[i].col;
        n_e_a[i]=edges[i].data;
    }

    
}
#endif // JIT

} // jittor