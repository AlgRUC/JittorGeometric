'''
Description: 
Author: lusz
Date: 2024-06-18 19:50:15
'''
import jittor as jt
x=jt.array([-1, 0, 1])
# y=jt.float32(x).abs()
y=x.abs().sum()
print(x.dtype)
print(y)