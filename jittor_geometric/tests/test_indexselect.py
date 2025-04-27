'''
Author: lusz
Date: 2025-04-26 16:11:32
Description: 
'''
import jittor as jt
from jittor import nn
import sys,os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.ops import IndexSelect
def test_index_slect():
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x=jt.array([[3.0, 3.0, 3.0],[2.0, 2.0, 2.0],[1.0, 1.0, 1.0]])
    y=jt.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
    dim=1
    index=jt.array([0,2])
    output=IndexSelect(x,dim,index)
    print(output)
    print(y)
    loss = nn.BCELoss()
    loss_var = loss(output, y)
    di = jt.grad(loss_var, [x])
    print("Input Variable Gradient:", di)

def test_performance_index_select():
    size=10000
    dim=1
    index_size=5000
    repeat=10
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    x = jt.rand((size, size))
    index = jt.randint(0, size, (index_size,))
    
    # 热身（Warm-up）
    _ = IndexSelect(x, dim, index)
    jt.sync_all()  # 确保执行完

    # 正式计时
    start = time.time()
    for _ in range(repeat):
        output = IndexSelect(x, dim, index)
        jt.sync_all()
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"IndexSelect performance test (size={size}, index_size={index_size}")
    print(f"Average time over {repeat} runs: {avg_time:.6f} seconds\n")



test_index_slect()
test_performance_index_select()