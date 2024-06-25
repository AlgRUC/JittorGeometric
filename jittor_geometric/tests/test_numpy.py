'''
Description: 
Author: lusz
Date: 2024-06-24 14:22:39
'''
import numpy as np
import jittor as jt
a=np.array([0,1,2])
b=jt.array(a)
print(a)
print(b)
c=np.matrix([[1,2],[3,4]])
print(c)
# d=jt.array(c) 这个不行
d=np.array(c)
e=jt.array(d)  # 这个可以
print(e)