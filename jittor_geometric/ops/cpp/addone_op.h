/*
 * @Author: lusz
 * @Date: 2024-06-21 10:19:29
 * @Description: 
 */

#pragma once
#include "op.h"
#include <immintrin.h>
namespace jittor {

struct AddoneOp : Op {
    Var* output;
    Var* outputVar;
    Var* inputVar;
    float64 weight;
    int feat_size;
    NanoString myType;
    AddoneOp(Var* outputVar_,Var* inputVar_,float64 weight_,int feat_size_,NanoString dtype=ns_float32);
    const char* name() const override { return "addone"; }
    DECLARE_jit_run;
};

} // jittor