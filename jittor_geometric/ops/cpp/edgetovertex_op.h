/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-28 17:08:33
 */

#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct EdgetovertexOp : Op {
    Var* x;
    Var* outputVar;
    Var* indices;
    Var* offset;
    Var* output;
    int flag;
    EdgetovertexOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_, int flag_);
    const char* name() const override { return "edgetovertex"; }
    DECLARE_jit_run;
};

} // jittor