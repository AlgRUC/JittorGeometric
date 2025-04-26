'''
Author: lusz
Date: 2025-04-26 16:11:32
Description: 
'''
import jittor as jt
from jittor import nn
import sys,os
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

test_index_slect()