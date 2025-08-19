import jittor as jt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from jittor_geometric.ops import getweight
jt.flags.lazy_execution = 0  # 关闭懒执行

#python jittor_geometric/tests/test_getweight.py
def test_getweight():
    """
    Test the getweight function for computing symmetric normalized edge weights.
    
    Graph structure:
        0 → 1, 0 → 2
        1 → 2
        2 → 3

    === CSC Representation ===
    column_offset: [0, 1, 2, 3, 4]  # 4 columns for nodes 0,1,2,3
    row_indices:   [0, 0, 1, 2]     # source nodes
    """
    # CSC inputs
    column_offset = jt.array([0, 1, 2, 3, 4], dtype="int32")
    row_indices   = jt.array([0, 0, 1, 2], dtype="int32")
    dst           = jt.array([0, 1, 2, 3], dtype="int32")
    src           = jt.array([0, 1, 2, 3], dtype="int32")
    vtx_size      = 4

    # Compute weights
    edge_weight = getweight(vtx_size, dst, src, column_offset, row_indices)

    # Output
    print("=== CSC Inputs ===")
    print("column_offset:", column_offset.numpy().tolist())
    print("row_indices:  ", row_indices.numpy().tolist())
    print("dst:          ", dst.numpy().tolist())
    print("src:          ", src.numpy().tolist())

    print("\n=== Computed Edge Weights ===")
    print(edge_weight.numpy().tolist())


    expected_weights = [
    1 / (jt.sqrt(2.0 * 1.0)),  # edge 0->0
    1 / (jt.sqrt(2.0 * 1.0)),  # edge 0->1
    1 / (jt.sqrt(1.0 * 1.0)),  # edge 1->2
    1 / (jt.sqrt(1.0 * 1.0)),  # edge 2->3
]

    expected_weights_values = [w.item() for w in expected_weights]

    print("\n=== Expected Edge Weights ===")
    print(expected_weights_values)

if __name__ == "__main__":
    test_getweight()
