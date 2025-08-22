/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-11-03 15:03:08
 */
#pragma once
#include "op.h"

#include "cusparse.h"
namespace jittor {

struct DistspmmOp : Op {
    Var* x;
    Var* outputVar;
    Var* col_indices;
    Var* row_offset;
    Var* value;
    NanoString dtype;
    Var* output;
    int A_row;
    int A_col;
    bool trans_A;
    bool trans_B;
    DistspmmOp(Var* outputVar_, Var* x_, Var* col_indices_,Var* value_,Var* row_offset_,int A_row_,int A_col_,bool trans_A_,bool trans_B_);
    const char* name() const override { return "distspmm"; }
    DECLARE_jit_run;
};

} // jittor