'''
Description: 
Author: lusz
Date: 2024-11-05 16:25:39
'''
import jittor as jt
import os
current_file_path = os.path.abspath(__file__)
test_path = os.path.dirname(current_file_path)
module_path = os.path.dirname(test_path)

src = os.path.join(module_path, "ops/cpp/spmmcsr_op.cc")
header = os.path.join(module_path, "ops/cpp/spmmcsr_op.h")
def test_spmm_csr():
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x=jt.array([[3.0, 2.0, 1.0],[3.0, 2.0, 1.0],[3.0, 2.0, 1.0]])
    col_indices=jt.array([0,1,1,2],dtype='int64')
    row_offset=jt.array([0,2,3,4],dtype='int64')
    csr_weight=jt.array([3.0,1.0,4.0,2.0], dtype='float32')
    output=jt.zeros((3,3), dtype='float32')
    spmmcsr_op = jt.compile_custom_ops((src, header))
    print(output)
    spmmcsr_op.spmmcsr(output,x,col_indices,csr_weight,row_offset,3,3).fetch_sync()
    print(output)

test_spmm_csr()