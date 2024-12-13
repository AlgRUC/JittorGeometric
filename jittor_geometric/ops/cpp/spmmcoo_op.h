/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-11-10 21:15:59
 */
#pragma once
#include "op.h"
#include "cusparse.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include "helper_cuda.h"
namespace jittor {

struct SpmmcooOp : Op {
    Var* x;
    Var* outputVar;
    Var* row_indices;
    Var* col_indices;
    Var* value;
    Var* output;
    int A_row;
    int A_col;
    SpmmcooOp(Var* outputVar_, Var* x_, Var* row_indices_,Var* col_indices_,Var* value_,int A_row,int A_col);
    const char* name() const override { return "spmmcoo"; }
    DECLARE_jit_run;
};

} // jittor