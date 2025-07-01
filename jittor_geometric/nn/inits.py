import math

import jittor as jt
from jittor import init


def uniform(size, var):
    if var is not None:
        bound = 1.0 / math.sqrt(size)
        init.uniform_(var, -bound, bound)


def kaiming_uniform(var, fan, a):
    if var is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        init.uniform_(var, -bound, bound)


def glorot(var):
    if var is not None:
        stdv = math.sqrt(6.0 / (var.size(-2) + var.size(-1)))
        init.uniform_(var, -stdv, stdv)

def glorot_orthogonal(var, scale):  
    if var is not None:  
        # 步骤1：正交初始化  
        # Jittor目前没有直接的正交初始化，需要自己实现  
        rows = var.size(-2)  
        cols = var.size(-1)  
        
        # 创建随机矩阵  
        flattened = var.view(rows, cols)  
        if rows < cols:  
            flattened = flattened.transpose((1, 0))  
            
        # QR分解实现正交初始化  
        q, r = jt.linalg.qr(jt.randn((max(rows, cols), max(rows, cols))))  
        # 处理符号以确保结果确定性  
        d = jt.diag(r)  
        ph = jt.nn.sign(d)  
        q *= ph  
        
        if rows < cols:  
            q = q.transpose((1, 0))  
        
        var = q[:rows, :cols]  
        
        scale /= ((var.size(-2) + var.size(-1)) * var.var())  
        
        var *= math.sqrt(scale)  
        
    return var 

def xavier_normal(var):
    if var is not None:
        stdv = math.sqrt(2.0 / (var.size(-2) + var.size(-1)))
        init.gauss_(var, mean=0.0, std=stdv)

def zeros(var):
    if var is not None:
        init.constant_(var, 0)


def ones(var):
    if var is not None:
        init.constant_(var, 1)


def normal(var, mean, std):
    if var is not None:
        var.assign(jt.normal(mean, std, size=var.size))


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
