/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-11-03 15:02:40
 */
#include "var.h"
#include "distspmm_op.h"
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



static inline cusparseOperation_t get_trans_type(bool trans) {
    if (trans) return CUSPARSE_OPERATION_TRANSPOSE;
    else return CUSPARSE_OPERATION_NON_TRANSPOSE;
}

#ifndef JIT


DistspmmOp::DistspmmOp(Var* outputVar_, Var* x_, Var* col_indices_,Var* value_,Var* row_offset_,int A_row_,int A_col_,bool trans_A_,bool trans_B_)
    : outputVar(outputVar_), x(x_), col_indices(col_indices_), value(value_),row_offset(row_offset_),A_row(A_row_),A_col(A_col_),trans_A(trans_A_),trans_B(trans_B_){
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0); 
    flags.set(NodeFlags::_manual_set_vnbb);
    ASSERT(x->dtype().is_float() && outputVar->dtype().is_float()) << "type of two inputs should be the same";
    output = create_output(nullptr, x->dtype());
}

void DistspmmOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
    add_jit_define(jk, "Tint", col_indices->dtype());
}


#else // JIT


void DistspmmOp::jit_run() {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseHandle_t     handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle) );
    // cusparseHandle_t & handle = cusparse_handle;
    const auto& xs = x->shape; // 密集矩阵维度
    const auto& vs = value->shape; 
    const auto& os = outputVar->shape;
    // ASSERT(xs==os)<<"matrix A and matrix C size not match";
    // ASSERT(A_col==xs[0])<<"matrix A and matrix B size not match";
    auto dtype_A = get_dtype(value->dtype());
    auto dtype_B = get_dtype(x->dtype());
    auto dtype_C = get_dtype(outputVar->dtype());
    auto dtype_index = get_index_dtype(col_indices->dtype());
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_row, A_col, vs[0], row_offset->ptr<Tint>(), col_indices->ptr<Tint>(), value->ptr<T>(), dtype_index, dtype_index, CUSPARSE_INDEX_BASE_ZERO, dtype_A) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, xs[0], xs[1], xs[1], x->ptr<T>(), dtype_B, CUSPARSE_ORDER_ROW) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, os[0], os[1],os[1], outputVar->ptr<T>(), dtype_C, CUSPARSE_ORDER_ROW) );
    float alpha = 1.0f;
    float beta  = 0.0f;
    // CHECK_CUSPARSE( cusparseSpMM_bufferSize(
    //                              handle,
    //                              CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                              CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
    //                              CUSPARSE_SPMM_CSR_ALG2, &bufferSize) );
    // CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )


    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        get_trans_type(trans_A),
        get_trans_type(trans_B),
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, &bufferSize) );
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );

    CHECK_CUSPARSE( cusparseSpMM(handle, get_trans_type(trans_A), get_trans_type(trans_B), &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2 , dBuffer));  //CUSPARSE_MM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG2 , CUSPARSE_SPMM_COO_ALG4
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseCreate(&handle) )
}
#endif // JIT

} // jittor