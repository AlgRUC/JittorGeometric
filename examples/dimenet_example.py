import jittor as jt
from jittor import Var, nn
import os
import os.path as osp
from functools import partial
from math import pi as PI
from math import sqrt
import sympy as sym
from typing import Callable, Dict, Optional, Tuple, Union

import math
import sympy as sym
from scipy import special as sp
from scipy.optimize import brentq
import numpy as np
import jittor
from jittor.nn import Embedding, Linear

from jittor_geometric.data import Dataset, download_url, CSC, CSR
from jittor_geometric.nn.inits import xavier_normal, glorot_orthogonal
from jittor_geometric.typing import OptVar, SparseVar
from jittor_geometric.utils import scatter
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.ops import from_nodes,to_nodes
from jittor_geometric.ops.repeat_interleave import repeat_interleave

qm9_target_dict: Dict[int, str] = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}

def radius_graph(pos, batch, r):  
    n = pos.size(0)  
    
    node_expanded = pos.unsqueeze(1).expand(n, n, 3)  
    node_pairwise_diff = node_expanded - pos.unsqueeze(0).expand(n, n, 3)  
    distances = jt.norm(node_pairwise_diff, dim=2)  

    batch_expanded = batch.unsqueeze(1).expand(n, n)  
    batch_pairs = batch_expanded == batch_expanded.t()  
    
    # 添加自环过滤  
    not_self_loops = 1 - jt.init.eye(n, dtype=jt.bool)  
    
    # 组合所有条件  
    mask = batch_pairs & (distances < r) & not_self_loops  

    edge_index = jt.nonzero(mask)  
    edge_index = edge_index.t()  
    return edge_index

def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

def spherical_bessel_formulas(n):
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis

def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * math.factorial(k - abs(m)) /
            (4 * np.pi * math.factorial(k + abs(m))))**0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    r"""Helper function to calculate Y_l^m."""
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify(
                    (1 - 2 * i) * P_l_m[i - 1][i - 1] * (1 - z**2)**0.5)
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, k):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if not isinstance(P_l_m[i][j], int):
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                                         y,
                                         sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                                         y,
                                         sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]
    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m

def swish(x):
    return x * jt.sigmoid(x)

class Envelope(jt.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def execute(self, x: jt.Var) -> jt.Var:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(jt.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = jt.Var(num_radial)

        self.reset_parameters()

    def reset_parameters(self):
        # with jt.no_grad():
        self.freq = jt.arange(1, self.freq + 1).float().mul_(PI)
        # self.freq.requires_grad_()

    def execute(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()

class SphericalBasisLayer(jt.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': jt.sin, 'cos': jt.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: jt.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def execute(self, dist: jt.Var, angle: jt.Var, idx_kj: jt.Var) -> jt.Var:
        dist = dist / self.cutoff
        rbf = jt.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = jt.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(jt.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.uniform_(-sqrt(3), sqrt(3))
        xavier_normal(self.lin.weight)
        xavier_normal(self.lin_rbf.weight)

    def execute(self, x: jt.Var, rbf: jt.Var, i: jt.Var, j: jt.Var) -> jt.Var:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(jt.concat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(jt.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal(self.lin1.weight)
        self.lin1.bias.fill_(0)
        xavier_normal(self.lin2.weight)
        self.lin2.bias.fill_(0)

    def execute(self, x: jt.Var) -> jt.Var:
        return x + self.act(self.lin2(self.act(self.lin1(x))))

def matrix_multiply_jittor(sbf, x_kj, W):  
    """  
    将einsum('wj,wl,ijl->wi', sbf, x_kj, W)转换为普通矩阵乘法  
    
    参数:  
    sbf: shape [w, j]  
    x_kj: shape [w, l]  
    W: shape [i, j, l]  
    
    返回:  
    result: shape [w, i]  
    """  
    # 获取维度  
    w = sbf.size(0)  # 批次大小  
    i = W.size(0)    # 输出特征维度  
    j = sbf.size(1)  # 第一个特征维度  
    l = x_kj.size(1) # 第二个特征维度  
    
    # 1. 重塑 W 为 [i, j*l]  
    W_flat = W.reshape((i, j*l))  # [i, j*l]  
    
    # 2. 计算 sbf 和 x_kj 的外积  
    # 扩展维度  
    sbf_expanded = sbf.unsqueeze(-1)     # [w, j, 1]  
    x_kj_expanded = x_kj.unsqueeze(1)    # [w, 1, l]  
    # 计算外积  
    outer_product = sbf_expanded * x_kj_expanded  # [w, j, l]  
    
    # 3. 重塑外积结果为 [w, j*l]  
    outer_flat = outer_product.reshape((w, j*l))  # [w, j*l]  
    
    # 4. 矩阵乘法得到最终结果 [w, i]  
    result = jt.nn.matmul(outer_flat, W_flat.transpose((1,0)))  # [w, i]  
    
    return result

class InteractionBlock(jt.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = jt.Var(
            jt.empty(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = jt.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = jt.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.fill_(0)
        self.W.gauss_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def execute(self, x: jt.Var, rbf: jt.Var, sbf: jt.Var, idx_kj: jt.Var,
                idx_ji: jt.Var) -> jt.Var:
        
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = matrix_multiply_jittor(sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(jt.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        assert output_initializer in {'zeros', 'xavier_normal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = jt.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal(self.lin_rbf.weight)
        for lin in self.lins:
            xavier_normal(lin.weight)
            lin.bias.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.fill_(0)
        elif self.output_initializer == 'xavier_normal':
            xavier_normal(self.lin.weight)

    def execute(self, x: jt.Var, rbf: jt.Var, i: jt.Var,
                num_nodes: Optional[int] = None) -> jt.Var:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(jt.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        assert output_initializer in {'zeros', 'xavier_normal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = jt.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal(self.lin_rbf.weight)
        xavier_normal(self.lin_up.weight)
        for lin in self.lins:
            xavier_normal(lin.weight)
            lin.bias.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.fill_(0)
        elif self.output_initializer == 'xavier_normal':
            xavier_normal(self.lin.weight)

    def execute(self, x: jt.Var, rbf: jt.Var, i: jt.Var,
                num_nodes: Optional[int] = None) -> jt.Var:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)

def triplets(  
    edge_index: jt.Var,  
    num_nodes: int,  
) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:  
    row, col = edge_index  # j->i  
    
    edge_weight = jt.ones(edge_index.shape[1])  
    
    csr = cootocsr(edge_index, edge_weight, num_nodes)  
    csc = cootocsc(edge_index, edge_weight, num_nodes)  
    
    out_degrees = scatter(edge_weight, row, dim=0, dim_size=num_nodes)  
    
    num_triplets = out_degrees[row].int()  
    
    idx_j = repeat_interleave(row, num_triplets, dim=0)  
    idx_i = repeat_interleave(col, num_triplets, dim=0)  
    
    idx_k = from_nodes(csc=csc, nodes=row) 
    
    edge_id = jt.arange(edge_index.shape[1])  
    idx_ji = repeat_interleave(edge_id, num_triplets, dim=0)  
    
    edge_id_map = -jt.ones(num_nodes * num_nodes, dtype=jt.int64)  
    edge_id_map[row * num_nodes + col] = edge_id  
    idx_kj = edge_id_map[idx_k * num_nodes + idx_j]  
    
    mask = idx_i != idx_k  
    idx_i = idx_i[mask]  
    idx_j = idx_j[mask]  
    idx_k = idx_k[mask]  
    idx_kj = idx_kj[mask]  
    idx_ji = idx_ji[mask]  
    
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

class DimeNet(jt.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"xavier_normal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[ Callable] = swish,
        output_initializer: str = 'xavier_normal',
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = jt.nn.ModuleList([
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = jt.nn.ModuleList([
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    @classmethod
    def from_qm9_pretrained(  
        cls,  
        root: str,  
        dataset: Dataset,  
        target: int,  
    ) -> Tuple['DimeNet', Dataset, Dataset, Dataset]:  # pragma: no cover  
        """返回在QM9数据集上预训练的DimeNet模型"""  
        assert target >= 0 and target <= 12 and not target == 4  

        root = osp.expanduser(osp.normpath(root))  
        path = osp.join(root, 'pretrained_dimenet', qm9_target_dict[target])  

        os.makedirs(path, exist_ok=True)  
        
        # 修改为下载pickle文件  
        model_path = osp.join(path, 'best_model.pkl')  
        if not osp.exists(model_path):  
            download_url(f'{cls.url}/{qm9_target_dict[target]}/best_model.pkl', path)  

        # 创建模型实例  
        model = cls(  
            hidden_channels=128,  
            out_channels=1,  
            num_blocks=6,  
            num_bilinear=8,  
            num_spherical=7,  
            num_radial=6,  
            cutoff=5.0,  
            envelope_exponent=5,  
            num_before_skip=1,  
            num_after_skip=2,  
            num_output_layers=3,  
        )  

        # 加载预训练参数  
        with open(model_path, 'rb') as f:  
            state = pickle.load(f)  

        # 复制参数到模型  
        # RBF层  
        model.rbf.freq = state['rbf_layer_freq']  
        
        # 嵌入层  
        model.emb.emb.weight = state['emb_block_embeddings']  
        model.emb.lin_rbf.weight = state['emb_block_dense_rbf_kernel']  
        model.emb.lin_rbf.bias = state['emb_block_dense_rbf_bias']  
        model.emb.lin.weight = state['emb_block_dense_kernel']  
        model.emb.lin.bias = state['emb_block_dense_bias']  

        # 输出块  
        for i, block in enumerate(model.output_blocks):  
            block.lin_rbf.weight = state[f'output_blocks_{i}_dense_rbf_kernel']  
            for j, lin in enumerate(block.lins):  
                lin.weight = state[f'output_blocks_{i}_dense_layers_{j}_kernel']  
                lin.bias = state[f'output_blocks_{i}_dense_layers_{j}_bias']  
            block.lin.weight = state[f'output_blocks_{i}_dense_final_kernel']  

        # 交互块  
        for i, block in enumerate(model.interaction_blocks):  
            block.lin_rbf.weight = state[f'int_blocks_{i}_dense_rbf_kernel']  
            block.lin_sbf.weight = state[f'int_blocks_{i}_dense_sbf_kernel']  
            block.lin_kj.weight = state[f'int_blocks_{i}_dense_kj_kernel']  
            block.lin_kj.bias = state[f'int_blocks_{i}_dense_kj_bias']  
            block.lin_ji.weight = state[f'int_blocks_{i}_dense_ji_kernel']  
            block.lin_ji.bias = state[f'int_blocks_{i}_dense_ji_bias']  
            block.W = state[f'int_blocks_{i}_bilinear']  
            
            # layers before skip  
            for j, layer in enumerate(block.layers_before_skip):  
                layer.lin1.weight = state[f'int_blocks_{i}_before_skip_{j}_dense1_kernel']  
                layer.lin1.bias = state[f'int_blocks_{i}_before_skip_{j}_dense1_bias']  
                layer.lin2.weight = state[f'int_blocks_{i}_before_skip_{j}_dense2_kernel']  
                layer.lin2.bias = state[f'int_blocks_{i}_before_skip_{j}_dense2_bias']  
            
            block.lin.weight = state[f'int_blocks_{i}_final_before_skip_kernel']  
            block.lin.bias = state[f'int_blocks_{i}_final_before_skip_bias']  
            
            # layers after skip  
            for j, layer in enumerate(block.layers_after_skip):  
                layer.lin1.weight = state[f'int_blocks_{i}_after_skip_{j}_dense1_kernel']  
                layer.lin1.bias = state[f'int_blocks_{i}_after_skip_{j}_dense1_bias']  
                layer.lin2.weight = state[f'int_blocks_{i}_after_skip_{j}_dense2_kernel']  
                layer.lin2.bias = state[f'int_blocks_{i}_after_skip_{j}_dense2_bias']  

        # 数据集分割（保持原有逻辑）  
        random_state = np.random.RandomState(seed=42)  
        perm = jt.array(random_state.permutation(np.arange(130831)))  
        perm = perm.long()  
        train_idx = perm[:110000]  
        val_idx = perm[110000:120000]  
        test_idx = perm[120000:]  

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def execute(
        self,
        z: jt.Var,
        pos: jt.Var,
        batch: OptVar = None,
    ) -> jt.Var:
        r"""Forward pass.

        Args:
            z (torch.jt.Var): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.jt.Var): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.jt.Var, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist =  (pos[i] - pos[j]).sqr().sum(dim=1).sqrt()
 

        # Calculate angles.
        pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = jt.misc.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = jt.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        
        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce='sum')