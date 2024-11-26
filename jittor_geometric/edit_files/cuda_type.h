/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   cuda_type.h
 * Author: wangqg
 *
 * Created on October 25, 2021, 9:39 AM
 */

#ifndef CUDA_TYPE_H
#define CUDA_TYPE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// #define CUDA_ENABLE 1
typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS = 512;
const int CUDA_NUM_BLOCKS = 128;
const int CUDA_NUM_THREADS_SOFTMAX = 32;
const int CUDA_NUM_BLOCKS_SOFTMAX = 512;
}
#ifdef __cplusplus

#endif

#endif /* CUDA_TYPE_H */
