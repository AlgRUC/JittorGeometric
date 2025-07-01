/*
 * @Author: lusz
 * @Date: 2024-06-20 21:40:53
 * @Description: 
 */
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct CootocsrOp : Op {
    Var* column_indices;
    Var* row_offset;
    Var* csr_edge_weight; // CSR

    Var* edge_index;
    Var* coo_edge_weight;// COO

    Var* output;
    int v_num;

    CootocsrOp(Var* edge_index_,Var* coo_edge_weight_,Var* column_indices_,Var* row_offset_,Var* csr_edge_weight_,int v_num_);
    const char* name() const override { return "cootocsr"; }
    DECLARE_jit_run;
};
} // jittor