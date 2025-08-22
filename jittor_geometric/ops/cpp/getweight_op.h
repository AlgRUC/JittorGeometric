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

#include <cmath>  
#include <cstdlib> 
#include <thread>


namespace jittor {


    struct GetweightOp : Op {

        Var* output;

  

        int vtx_size;
        Var* csc_layer_dst;
        Var* csc_layer_src;
        Var* csc_layer_column_offset;
        Var* csc_layer_row_indices;

        Var* csc_layer_edge_weight;



       
        GetweightOp(int vtx_size_, Var* csc_layer_dst_, Var* csc_layer_src_, Var* csc_layer_column_offset_, Var* csc_layer_row_indices_, Var* csc_layer_edge_weight_);

        const char* name() const override { return "getweight"; }
        DECLARE_jit_run;

    };





} // namespace jittor
