import jittor as jt
import os
module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cootocsr.cc")
header = os.path.join(module_path, "cootocsr.h")

def test_coo_to_csr():
    jt.flags.use_cuda = 0
    jt.flags.lazy_execution = 0

    # Constructing a more complex example
    edge_index = jt.array([[0, 0, 1, 1, 2], [1, 2, 2, 3, 3]])
    edge_weight = jt.array([1.0, 2.0, 3.0, 4.0, 5.0])
    csr_edge_weight = jt.randn(5)
    column_indices = jt.zeros((5,), dtype='int32')
    row_offset = jt.zeros((5,), dtype='int32')
    v_num = 4

    # Create and compile the custom op
    my_op = jt.compile_custom_op(header, src, "cootocsr", warp=False)
    my_op(edge_index, edge_weight, column_indices, row_offset, csr_edge_weight, v_num, 'float32').fetch_sync()
    
    print("CSR Edge Weight:", csr_edge_weight)
    print("Column Indices:", column_indices)
    print("Row Offset:", row_offset)

test_coo_to_csr()