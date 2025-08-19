/*
 * @Description: 
 * @Author: liuyy
 */
#pragma once
#include "op.h"
#include <immintrin.h>
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
