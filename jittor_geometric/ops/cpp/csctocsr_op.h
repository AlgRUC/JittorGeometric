/*
 * @Description: 
 * @Author: liuyy
 */

#pragma once
#include "op.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
namespace jittor {

struct CsctocsrOp : Op {

    Var* output;

    Var* column_offset;
    Var* row_indices;
    Var* row_offset;
    Var* column_indices;
    
    Var* src;
    Var* dst;



    CsctocsrOp(Var* column_offset_,Var* row_indices_,Var* row_offset_,Var* column_indices_,Var* dst_, Var* src_);
    const char* name() const override { return "csctocsr"; }
    DECLARE_jit_run;
};

} // jittor