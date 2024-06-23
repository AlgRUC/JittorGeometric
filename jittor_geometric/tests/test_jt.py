'''
Description: 
Author: lusz
Date: 2024-06-23 19:10:45
'''
from jittor import jt
nvector = jt.NanoVector
nv = nvector()
nv.append(1)
nv.append(2)
nv.append(3)
print(nv)
# var_nv=jt.array(nv)
# print(var_nv)