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


    struct GpuinitcoOp : Op {

        Var* output;

  
        int dst_size;
        int fanout;
        Var* dst;
        Var* csc_layer_column_offset;//最终要填的
        Var* csc_global_column_offset;



       

        GpuinitcoOp(int dst_size_,int fanout_,Var* dst_,Var* csc_layer_column_offset_,Var* csc_global_column_offset_);

        const char* name() const override { return "gpuinitco"; }
        DECLARE_jit_run;

    };





} // namespace jittor
