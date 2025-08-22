import jittor as jt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.utils.num_nodes import maybe_num_nodes
from jittor_geometric.ops import gpuinitco

jt.flags.lazy_execution = 0  # 关闭懒执行

#python jittor_geometric/tests/test_gpuinitco.py


dst_size = 4
fanout_i = 2
csc_layer_dst = jt.array([0, 1, 2, 3], dtype="int32")
csc_global_column_offset = jt.array([0, 1, 3, 4, 5], dtype="int32")  # 全局 CSC 列偏移

csc_layer_column_offset = gpuinitco(dst_size, fanout_i, csc_layer_dst, csc_global_column_offset)

print("局部 CSC 列偏移:", csc_layer_column_offset.numpy().tolist())
# 输出示例: [0, 1, 2, 1, 1]
