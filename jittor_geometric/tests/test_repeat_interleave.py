import jittor as jt
import os.path as osp
import sys,os
current_dir = osp.dirname(osp.abspath(__file__))
root = osp.dirname(osp.dirname(current_dir))
sys.path.append(root)

from jittor_geometric.ops.repeat_interleave import repeat_interleave

y1= jt.array([1, 2, 3])
result1= repeat_interleave(y1, 2)
print(result1)

y2= jt.array([[1, 2], [3, 4]])
result2= repeat_interleave(y2, 2)
print(result2)

result3= repeat_interleave(y2, 3,1)
print(result3)

result4= repeat_interleave(y2, jt.array([1, 2]),0)
print(result4)

result4= repeat_interleave(y2, jt.array([1, 2]),1)
print(result4)

result4= repeat_interleave(y2, jt.array([1, 2]),-2)
print(result4)

y3= jt.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]])

# Test dim=1, repeats = [1, 2]
result5 = repeat_interleave(y3, jt.array([1, 2]), dim=1)
print("dim=1:", result5)

# Test dim=2, repeats = [2, 1, 3]
result6 = repeat_interleave(y3, jt.array([2, 1, 3]), dim=2)
print("dim=2:", result6)
