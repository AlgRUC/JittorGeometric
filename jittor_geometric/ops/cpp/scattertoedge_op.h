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

struct ScattertoedgeOp : Op {
    Var* x;
    Var* outputVar;
    Var* edge_weight;
    Var* indices;
    Var* offset;
    bool with_weight;
    NanoString dtype;
    Var* output;
    int flag;
    ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_,Var* edge_weight_,bool with_weight_,int flag_,NanoString dype_=ns_float32);
    ScattertoedgeOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_, bool with_weight_,int flag_ ,NanoString dtype_=ns_float32);
    const char* name() const override { return "scattertoedge"; }
    DECLARE_jit_run;
};

} // jittor