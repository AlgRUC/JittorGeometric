/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-11-10 21:16:17
 */
#include "var.h"
#include "spmmcoo_op.h"
using namespace std;

namespace jittor {

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}



#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

static inline cusparseIndexType_t get_index_dtype(NanoString dtype) {
    if (dtype == ns_int32) return CUSPARSE_INDEX_32I;
    if (dtype == ns_int64) return CUSPARSE_INDEX_64I;
    LOGf << "not support type" << dtype;
    return CUSPARSE_INDEX_32I;
}

static inline cudaDataType get_dtype(NanoString dtype) {
    if (dtype == ns_float32) return CUDA_R_32F;
    if (dtype == ns_float64) return CUDA_R_64F;
    if (dtype == ns_float16) return CUDA_R_16F;
    #ifndef IS_ROCM
    if (dtype == ns_bfloat16) return CUDA_R_16BF;
    #endif
    LOGf << "not support type" << dtype;
    return CUDA_R_32F;
}
#ifndef JIT

SpmmcooOp::SpmmcooOp(Var* outputVar_, Var* x_, Var* row_indices_,Var* col_indices_,Var* value_,int A_row_,int A_col_)
    : outputVar(outputVar_), x(x_),row_indices(row_indices_), col_indices(col_indices_), value(value_),A_row(A_row_),A_col(A_col_) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0); 
    flags.set(NodeFlags::_manual_set_vnbb);
    ASSERT(x->dtype().is_float() && outputVar->dtype().is_float()) << "type of two inputs should be the same";
    output = create_output(nullptr, x->dtype());
}

void SpmmcooOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
    add_jit_define(jk, "Tint", row_indices->dtype());
}

#else // JIT

void SpmmcooOp::jit_run() {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t     handle = NULL;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle) );
    const auto& xs = x->shape;
    const auto& vs = value->shape; 
    const auto& os = outputVar->shape;
    ASSERT(xs==os)<<"matrix A and matrix C size not match";
    ASSERT(A_col==xs[0])<<"matrix A and matrix B size not match";
    auto dtype_A = get_dtype(value->dtype());
    auto dtype_B = get_dtype(x->dtype());
    auto dtype_C = get_dtype(outputVar->dtype());
    auto dtype_index = get_index_dtype(col_indices->dtype());
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_row, A_col, vs[0], row_indices->ptr<Tint>(), col_indices->ptr<Tint>(), value->ptr<T>(), dtype_index, CUSPARSE_INDEX_BASE_ZERO, dtype_A ) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, xs[0], xs[1], xs[1], x->ptr<T>(), dtype_B, CUSPARSE_ORDER_ROW) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, os[0], os[1],os[1], outputVar->ptr<T>(), dtype_C, CUSPARSE_ORDER_ROW) );
    float alpha = 1.0f;
    float beta  = 0.0f;
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT , &bufferSize) );
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
    CHECK_CUDA( cudaFree(dBuffer) );
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) );
    CHECK_CUSPARSE(cusparseDestroy(handle) );
}
#endif // JIT

} // jittor