/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-23 16:06:10
 */

#pragma once
#include "op.h"
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
namespace jittor {

struct ToundirectedOp : Op {
    Var* output; 
    Var* edge_index;
    Var* edge_attr;
    Var* new_edge_index;
    Var* new_edge_attr;
    int num_edges;
    int num_nodes;
    NanoString dtype;
    ToundirectedOp(Var* edge_index_,Var* edge_attr_,int num_edges_,int num_nodes_,Var* new_edge_index_,Var* new_edge_attr_,NanoString dtype_=ns_float32);
    const char* name() const override { return "toundirected"; }
    DECLARE_jit_run;
};

} // jittor