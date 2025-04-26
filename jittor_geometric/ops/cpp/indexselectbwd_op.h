/*
 * @Author: lusz
 * @Date: 2025-04-26 22:34:26
 * @Description: 
 */
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct IndexselectbwdOp : Op {
    Var* x;
    Var* outputVar;
    int dim;
    Var* index;
    Var* output;
    IndexselectbwdOp(Var* outputVar_, Var* x_,int dim_,Var* index_);
    const char* name() const override { return "indexselectbwd"; }
    DECLARE_jit_run;
};

} // jittor