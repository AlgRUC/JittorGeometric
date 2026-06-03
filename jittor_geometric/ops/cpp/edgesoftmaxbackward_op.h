/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-04 16:16:31
 */


#pragma once
#include "op.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cstdlib>
#include <thread>
#include <math.h>
namespace jittor {

struct EdgesoftmaxbackwardOp : Op {
    Var* x;
    Var* outputVar;
    Var* y;
    Var* indices;
    Var* offset;
    Var* edge_weight;
    Var* output;
    EdgesoftmaxbackwardOp(Var* outputVar_, Var* x_,Var* y_, Var* indices_,Var* offset_);
    const char* name() const override { return "edgesoftmaxbackward"; }
    DECLARE_jit_run;
};

} // jittor