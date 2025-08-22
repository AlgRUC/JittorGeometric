import jittor as jt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from jittor_geometric.ops.csctocsr import csctocsr

jt.flags.lazy_execution = 0  # 关闭懒执行

#python jittor_geometric/tests/test_csctocsr.py
def test_csctocsr():
    """   
    图结构：
        0 → 1, 0 → 2
        1 → 3
        2 → 3

    === COO（坐标格式） ===
    src: [0, 0, 1, 2]
    dst: [1, 2, 3, 3]

    """

    # === CSC 格式 ===
    column_offset = jt.array([0, 0, 1, 2, 4], dtype="int32")
    row_indices   = jt.array([0, 0, 1, 2], dtype="int32")

    # 所有 dst 节点 ID
    dst = jt.array([0, 1, 2, 3], dtype="int32")
    # 所有 src 节点 ID
    src = jt.array([0, 1, 2, 3], dtype="int32")

    # 转换
    row_offset, column_indices = csctocsr(column_offset, row_indices, dst, src)

    # 输出
    print("=== 输入 CSC ===")
    print("column_offset:", column_offset.numpy().tolist())
    print("row_indices:  ", row_indices.numpy().tolist())
    print("dst:          ", dst.numpy().tolist())
    print("src:          ", src.numpy().tolist())

    print("\n=== 输出 CSR ===")
    print("row_offset:   ", row_offset.numpy().tolist())
    print("column_indices", column_indices.numpy().tolist())



if __name__ == "__main__":
    test_csctocsr()