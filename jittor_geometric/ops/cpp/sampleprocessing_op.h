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
#include <unordered_set>
#include <cstdlib>
#include <thread>
#include <cstddef>



namespace jittor {


    struct SampleprocessingOp : Op {

        Var* output;
  
        int fanout_i;
        Var* csc_layer_dst;
        Var* csc_layer_column_offset;
        Var* csc_layer_row_indices;
        Var* csc_global_column_offset;
        Var* csc_global_row_indices;
       

        SampleprocessingOp(int fanout_i_,Var* csc_layer_dst_,Var* csc_layer_column_offset_,Var* csc_layer_row_indices_,Var* csc_global_column_offset_,Var* csc_global_row_indices_);

        const char* name() const override { return "sampleprocessing"; }
        DECLARE_jit_run;
    };





} // namespace jittor
