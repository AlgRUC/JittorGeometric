/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-11-03 15:03:08
 */
#pragma once
#include "op.h"

#include "cusparse.h"
namespace jittor {

struct SpmmcsrOp : Op {
    Var* x;
    Var* outputVar;
    Var* col_indices;
    Var* row_offset;
    Var* value;
    NanoString dtype;
    Var* output;
    int A_row;
    int A_col;
    SpmmcsrOp(Var* outputVar_, Var* x_, Var* col_indices_,Var* value_,Var* row_offset_,int A_row,int A_col,NanoString dtype_=ns_float32);
    const char* name() const override { return "spmmcsr"; }
    DECLARE_jit_run;
};

} // jittor