import jittor as jt

header ="""
#pragma once
#include "op.h"

namespace jittor {

struct CustomOp : Op {
    Var* output;
    CustomOp(NanoVector shape, NanoString dtype=ns_float32);
    
    const char* name() const override { return "custom"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "custom_op.h"

namespace jittor {
#ifndef JIT
CustomOp::CustomOp(NanoVector shape, NanoString dtype) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(shape, dtype);
}

void CustomOp::jit_prepare(JK& jk) {
    // std::cout<<output->dtype()<<std::endl; float32
    add_jit_define(jk, "T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    for (index_t i=0; i<num; i++)
        x[i] = (T)i;
}
#else
// JIT_cuda
__global__ void kernel(index_t n, T *x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = (T)-i;
}

void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    int blockSize = 256;
    int numBlocks = (num + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(num, x);
}
#endif // JIT_cpu
#endif // JIT

} // jittor
"""

my_op = jt.compile_custom_op(header, src, "custom", warp=False)

# run cpu version
jt.flags.use_cuda = 0
a = my_op([3,4,5], 'float').fetch_sync()
assert (a.flatten() == range(3*4*5)).all()
print(a)

if jt.compiler.has_cuda:
    # run cuda version
    jt.flags.use_cuda = 1
    a = my_op([3,4,5], 'float').fetch_sync()
    print(a)
    assert (-a.flatten() == range(3*4*5)).all()