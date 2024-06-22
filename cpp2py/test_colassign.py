import jittor as jt
from jittor import nn

header ="""
#pragma once
#include "op.h"
#include <immintrin.h>
#include <cstdlib>
#include <thread>
namespace jittor {

struct ColassignOp : Op {
    Var* x;
    Var* result;
    int i;
    int row;
    int feature_dim;
    NanoString dtype;
    Var* output; //必须要一个output,暂时没找到解决办法
    ColassignOp(Var* result_,Var* x_,int i_,int row_, int feature_dim_,NanoString dtype_=ns_float32);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    const char* name() const override { return "colassign"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "var.h"
#include "ops/op_register.h"
#include "colassign_op.h"


namespace jittor {
//static auto make_colassign = get_op_info("colassign")
//  .get_constructor<VarPtr, Var*>();

#ifndef JIT
ColassignOp::ColassignOp(Var* result_,Var* x_,int i_,int row_, int feature_dim_,NanoString dtype_) : 
x(x_), result(result_),i(i_), row(row_),feature_dim(feature_dim_),dtype(dtype_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,dtype);
}

VarPtr ColassignOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    std::cout<<"1"<<std::endl;
    return dout;
}

void ColassignOp::jit_prepare(JK& jk) {
    //std::cout<<myType<<std::endl;
     add_jit_define(jk, "T", dtype);
}

#else // JIT
void ColassignOp::jit_run() {
    int max_threads = std::thread::hardware_concurrency();
    auto* __restrict__ r_ptr = result->ptr<T>();
    auto* __restrict__ x_ptr = x->ptr<T>();
    #pragma omp parallel for num_threads(max_threads) schedule(guided)
    for(int j=0;j<feature_dim;j++){
        r_ptr[i*feature_dim+j]=x_ptr[row*feature_dim+j];
    }
}
#endif // JIT

} // jittor
"""
def test_col():
    jt.flags.use_cuda = 0
    jt.flags.lazy_execution = 0

    # Constructing a more complex example
    x = jt.array([[0, 0, 1], [1, 2, 2]]).float32()
    result = jt.zeros((5,3))
    row=1
    i=3
    feature_dim=3
    # Create and compile the custom op
    my_op = jt.compile_custom_op(header, src, "colassign", warp=False)
    my_op(result,x,i,row,feature_dim,'float32').fetch_sync()
    print(result)
    y=jt.array([0, 0, 1, 1, 2, 2,1, 2, 2,1, 2, 2,1, 2, 2]).float32()
    re_result=jt.reshape(result,(1,15))
    # jt.reshape(y,(1,15))
    print(re_result)
    loss = nn.nll_loss(re_result,y)
    
    
test_col()