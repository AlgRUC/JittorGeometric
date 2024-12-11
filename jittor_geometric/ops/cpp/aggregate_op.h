/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:12
 */
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct AggregateOp : Op {
    Var* x;
    Var* outputVar;
    Var* indices;
    Var* offset;
    Var* weight;
    bool forward;
    Var* output;
    AggregateOp(Var* outputVar, Var* x_,Var* indices_,Var* offset_,Var* weight_,bool forward_);
    const char* name() const override { return "aggregate"; }
    DECLARE_jit_run;
};

} // jittor