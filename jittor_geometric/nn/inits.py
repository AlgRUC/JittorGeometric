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
