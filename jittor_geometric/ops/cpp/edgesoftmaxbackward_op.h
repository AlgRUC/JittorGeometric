/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-04 16:16:31
 */


#pragma once
#include "op.h"
#include <immintrin.h>
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