/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-21 14:14:12
 */
#pragma once
#include "op.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
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