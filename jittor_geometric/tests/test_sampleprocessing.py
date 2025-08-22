import jittor as jt
import os
import sys
import numpy as np
sys.path.append('/home/liuyy/jittor/JittorGeometric')
from jittor_geometric.ops import sampleprocessing


jt.flags.lazy_execution = 0  # 关闭懒执行
import jittor as jt
from jittor_geometric.ops import sampleprocessing  # 假设你已经编译好 op
#python jittor_geometric/tests/test_sampleprocessing.py


import jittor as jt
from jittor_geometric.ops import sampleprocessing

def test_sampleprocessing_larger_graph():
    """
    图结构（全局 CSC 表示）：
        节点 0 -> 1,2,3
        节点 1 -> 2,3,4
        节点 2 -> 0,5
        节点 3 -> 1,6
        节点 4 -> 2,7
        节点 5 -> 3,8
        节点 6 -> 4,9
        节点 7 -> 5
        节点 8 -> 6
        节点 9 -> 7

    局部 CSC 列偏移（已采样，fanout=3）
    """
    fanout_i = 3
    csc_layer_dst = jt.array([0,1,2,3,4,5,6,7,8,9], dtype="int32")

    csc_layer_column_offset = jt.array([0, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18], dtype="int32")


    csc_global_column_offset = jt.array([0, 3, 6, 8, 10, 12, 14, 16, 17, 18, 19], dtype="int32")
    csc_global_row_indices   = jt.array([1,2,3,2,3,4,0,5,1,6,2,7,3,8,4,9,5,6,7], dtype="int32")

    # 调用 sampleprocessing
    csc_layer_row_indices = sampleprocessing(fanout_i, csc_layer_dst, csc_layer_column_offset,
                                             csc_global_column_offset, csc_global_row_indices)

    # 输出
    print("=== 输入 ===")
    print("csc_layer_dst:", csc_layer_dst.numpy().tolist())
    print("csc_layer_column_offset:", csc_layer_column_offset.numpy().tolist())
    print("csc_global_column_offset:", csc_global_column_offset.numpy().tolist())
    print("csc_global_row_indices:", csc_global_row_indices.numpy().tolist())

    print("\n=== 输出 ===")
    print("csc_layer_row_indices:", csc_layer_row_indices.numpy().tolist())

if __name__ == "__main__":
    test_sampleprocessing_larger_graph()




