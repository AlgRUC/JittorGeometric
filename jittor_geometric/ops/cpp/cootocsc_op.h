/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 20:20:26
 */
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

    Var* output;
    int v_num;

    CootocscOp(Var* edge_index_, Var* coo_edge_weight_, Var* row_indices_, Var* column_offset_, Var* csc_edge_weight_, int v_num_);
    const char* name() const override { return "cootocsc"; }
    DECLARE_jit_run;
};

} // jittor