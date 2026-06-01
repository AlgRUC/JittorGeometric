/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-23 16:06:10
 */

#pragma once
#include "op.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
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