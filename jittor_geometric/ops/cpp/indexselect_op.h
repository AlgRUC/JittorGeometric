/*
 * @Description: 
 * @Author: lusz
 * @Date: 2025-04-26 14:55:20
 */
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct IndexselectOp : Op {
    Var* x;
    Var* outputVar;
    int dim;
    Var* index;
    Var* output;
    IndexselectOp(Var* outputVar_, Var* x_,int dim_,Var* index_);
    const char* name() const override { return "indexselect"; }
    DECLARE_jit_run;
};

} // jittor