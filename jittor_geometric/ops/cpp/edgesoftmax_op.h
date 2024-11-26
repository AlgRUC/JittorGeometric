/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-03 13:50:18
 */

#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
#include <math.h>
namespace jittor {

struct EdgesoftmaxOp : Op {
    Var* x;
    Var* outputVar;
    Var* indices;
    Var* offset;
    Var* edge_weight;
    NanoString dtype;
    Var* output;
    EdgesoftmaxOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,NanoString dype_=ns_float32);
    const char* name() const override { return "edgesoftmax"; }
    DECLARE_jit_run;
};

} // jittor