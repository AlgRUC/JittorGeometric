# coding=utf-8

"""
Spherical Message Passing for 3D Graph Networks
https://arxiv.org/abs/2102.05013?context=cs
"""


import sys
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import jittor
from jittor import Var, nn, Module
from jittor.nn import Linear, Embedding
from math import sqrt
from math import pi as PI
import sympy as sym
sys.path.append("../../../")
from jittor_geometric.nn.inits import glorot, zeros
from jittor_geometric.typing import Adj, OptVar
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight


def scatter(src, batch, dim=0, reduce='add', dim_size=None):
    dim_0 = src.shape[1]
    num_0 = batch.shape[0]
    if dim_size is None:
        dim_size = jittor.int(batch.max()).item() + 1
    x = jittor.zeros(dim_size, dim_0)
    batch = batch.repeat_interleave(dim_0).reshape(num_0, dim_0)
    x = jittor.scatter(x, dim, batch, src, reduce=reduce)
    return x

def swish(x):
    import jittor
    #return x * torch.sigmoid(x)
    return x * jittor.sigmoid(x)


class Envelope(jittor.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def execute(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(jittor.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = jittor.Var(num_radial)
        #self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        #self.reset_parameters()

    def reset_parameters(self):
        self.freq = jittor.arange(1, self.freq + 1).float().mul_(PI)

    def execute(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': jittor.sin, 'cos':jittor.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: jittor.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def execute(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = jittor.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = jittor.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': jittor.sin, 'cos': jittor.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: jittor.zeros_like(x) + jittor.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def execute(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = jittor.stack([f(dist) for f in self.bessel_funcs], dim=1)
        """
        print(angle, phi)
        print(jittor.zeros_like(angle).shape)
        print(jittor.zeros_like(phi).shape)
        tmp = jittor.zeros_like(angle) + jittor.zeros_like(phi)
        """
        #+ self.sph1(0,0)
        cbf = jittor.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out

class emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def execute(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb

class ResidualLayer(jittor.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        self.lin1.bias.fill_(0)
        glorot(self.lin2.weight)
        self.lin2.bias.fill_(0)

    def execute(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(jittor.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True, use_extra_node_feature=False):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        self.use_extra_node_feature = use_extra_node_feature
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(jittor.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        if self.use_extra_node_feature:
            self.lin = Linear(5 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        #self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.uniform_(-sqrt(3), sqrt(3))
        #self.lin_rbf_0.reset_parameters()
        #self.lin.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.lin_rbf_0.weight)
        glorot(self.lin_rbf_1.weight)

    def execute(self, x, node_feature, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        if node_feature != None and self.use_extra_node_feature:
            x = jittor.concat((x, node_feature), 1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(jittor.concat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


# 改了这里
class update_e(jittor.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = jittor.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = jittor.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_rbf1.weight)
        glorot(self.lin_rbf2.weight)
        glorot(self.lin_sbf1.weight)
        glorot(self.lin_sbf2.weight)
        glorot(self.lin_t1.weight)
        glorot(self.lin_t2.weight)

        glorot(self.lin_kj.weight)
        self.lin_kj.bias.fill_(0)
        glorot(self.lin_ji.weight)
        self.lin_ji.bias.fill_(0)

        glorot(self.lin_down.weight)
        glorot(self.lin_up.weight)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot(self.lin.weight)
        self.lin.bias.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot(self.lin_rbf.weight)

    def execute(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


# 重写 radius graph
def radius_graph(pos, batch, r):
    n = pos.size(0)
    #device = node.device

    # 计算所有节点对之间的距离
    node_expanded = pos.unsqueeze(1).expand(n, n, 3)
    node_pairwise_diff = node_expanded - pos.unsqueeze(0).expand(n, n, 3)
    distances = jittor.norm(node_pairwise_diff, dim=2)

    # 创建批次索引对
    batch_expanded = batch.unsqueeze(1).expand(n, n)
    batch_pairs = batch_expanded == batch_expanded.t()

    # 只保留批次内部的节点对
    mask = batch_pairs & (distances < r)

    # 获取边的索引
    edge_index = jittor.nonzero(mask)
    edge_index = edge_index.t()
    return edge_index

# 无需修改
def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)

# 无需修改
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


# 无需修改
def spherical_bessel_formulas(n):
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f


# 无需修改
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


# 无需修改
def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m))))**0.5


# 无需修改
def associated_legendre_polynomials(k, zero_m_only=True):
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
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m


# 无需修改
def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        x = sym.symbols('x')
        y = sym.symbols('y')
        S_m = [x*0]
        C_m = [1+0*x]
        # S_m = [0]
        # C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


# TODO: torsion. Need jittor CSC, scatter and etc.
def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False):
    """
    Compute the distance, angle, and torsion from geometric information.

    Args:
        pos (Tensor): Geometric information for every node in the graph.
        edge_index (Tensor): Edge index of the graph (shape: [2, num_edges]).
        num_nodes (int): Number of nodes in the graph.
        use_torsion (bool): If True, return distance, angle, and torsion. Default is False.

    Returns:
        Tuple: Distance, angle, and (optional) torsion, along with some useful indices.
    """
    j, i = edge_index  # j -> i

    dist = jittor.norm(pos[i] - pos[j], p=2, dim=-1)


    # （k -> j -> i）
    idx_i, idx_j, idx_k = [], [], []
    for tmp_index in range(i.shape[0]):
        tmp_idx_i = i[tmp_index]
        tmp_idx_j = j[tmp_index]
        tmp_idx_k = j[i == tmp_idx_j]
        tmp_num_triplets = tmp_idx_k.shape[0]
        if tmp_num_triplets == 0:
            continue
        idx_i.append(tmp_idx_i.repeat_interleave(tmp_num_triplets))
        idx_j.append(tmp_idx_j.repeat_interleave(tmp_num_triplets))
        idx_k.append(tmp_idx_k)
    idx_i = jittor.concat(idx_i)
    idx_j = jittor.concat(idx_j)
    idx_k = jittor.concat(idx_k)

    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # angle
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = jittor.sum(pos_ji * pos_jk, dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = jittor.norm(jittor.cross(pos_ji, pos_jk), dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = jittor.atan2(b, a)

    # torsion
    torsion = jittor.zeros_like(angle)

    return dist, angle, torsion, i, j, idx_k, idx_j
    # fix torsion
    if use_torsion:
        idx_batch = jittor.arange(len(idx_i), device=pos.device)
        idx_k_n = jittor.index_select(adj_t_row, 0, idx_j)
        num_triplets_t = jittor.tile(num_triplets, num_triplets)[mask]
        idx_i_t = jittor.misc.repeat_interleave(idx_i, num_triplets_t)
        idx_j_t = jittor.misc.repeat_interleave(idx_j, num_triplets_t)
        idx_k_t = jittor.misc.repeat_interleave(idx_k, num_triplets_t)
        idx_batch_t = jittor.misc.repeat_interleave(idx_batch, num_triplets_t)
        mask = idx_i_t != idx_k_n
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = jittor.norm(pos_ji, p=2, dim=-1)
        plane1 = jittor.cross(pos_ji, pos_j0)
        plane2 = jittor.cross(pos_ji, pos_jk)
        a = jittor.sum(plane1 * plane2, dim=-1)  # cos_angle * |plane1| * |plane2|
        b = jittor.sum(jittor.cross(plane1, plane2) * pos_ji, dim=-1) / dist_ji
        torsion1 = jittor.atan2(b, a)  # -pi to pi
        torsion1[torsion1 <= 0] += 2 * np.pi  # 0 to 2pi
        torsion = jittor.scatter(torsion1, idx_batch_t, reduce='min')

        return dist, angle, torsion, i, j, idx_k, idx_j
    else:
        return dist, angle, i, j, idx_k, idx_j

def swish(x):
    import jittor
    return x * jittor.sigmoid(x)


class Envelope(jittor.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def execute(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(jittor.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = jittor.Var(num_radial)
        #self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        #self.reset_parameters()

    def reset_parameters(self):
        self.freq = jittor.arange(1, self.freq + 1).float().mul_(PI)

    def execute(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': jittor.sin, 'cos':jittor.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: jittor.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def execute(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = jittor.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = jittor.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': jittor.sin, 'cos': jittor.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: jittor.zeros_like(x) + jittor.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def execute(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = jittor.stack([f(dist) for f in self.bessel_funcs], dim=1)
        """
        print(angle, phi)
        print(jittor.zeros_like(angle).shape)
        print(jittor.zeros_like(phi).shape)
        tmp = jittor.zeros_like(angle) + jittor.zeros_like(phi)
        """
        #+ self.sph1(0,0)
        cbf = jittor.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out

class emb(jittor.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def execute(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb

class ResidualLayer(jittor.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        self.lin1.bias.fill_(0)
        glorot(self.lin2.weight)
        self.lin2.bias.fill_(0)

    def execute(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(jittor.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True, use_extra_node_feature=False):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        self.use_extra_node_feature = use_extra_node_feature
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(jittor.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        if self.use_extra_node_feature:
            self.lin = Linear(5 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        #self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.uniform_(-sqrt(3), sqrt(3))
        #self.lin_rbf_0.reset_parameters()
        #self.lin.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.lin_rbf_0.weight)
        glorot(self.lin_rbf_1.weight)

    def execute(self, x, node_feature, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        if node_feature != None and self.use_extra_node_feature:
            x = jittor.concat((x, node_feature), 1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(jittor.concat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(jittor.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = jittor.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = jittor.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_rbf1.weight)
        glorot(self.lin_rbf2.weight)
        glorot(self.lin_sbf1.weight)
        glorot(self.lin_sbf2.weight)
        glorot(self.lin_t1.weight)
        glorot(self.lin_t2.weight)

        glorot(self.lin_kj.weight)
        self.lin_kj.bias.fill_(0)
        glorot(self.lin_ji.weight)
        self.lin_ji.bias.fill_(0)

        glorot(self.lin_down.weight)
        glorot(self.lin_up.weight)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot(self.lin.weight)
        self.lin.bias.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot(self.lin_rbf.weight)

    def execute(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(jittor.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = jittor.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        #glorot_orthogonal(self.lin_up.weight, scale=2.0)
        glorot(self.lin_up.weight)
        for lin in self.lins:
            glorot(lin.weight)
            lin.bias.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot(self.lin.weight)

    def execute(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(jittor.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def execute(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class SphereNet(jittor.nn.Module):
    r"""
         The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.

        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`)
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)

    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True, use_extra_node_feature=False, extra_node_feature_dim=1):
        super(SphereNet, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.use_extra_node_feature = use_extra_node_feature

        if use_extra_node_feature:
            self.extra_emb = Linear(extra_node_feature_dim, hidden_channels)

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features, use_extra_node_feature=use_extra_node_feature)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = jittor.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])

        self.update_es = jittor.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        self.update_us = jittor.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_extra_node_feature:
            self.extra_emb.reset_parameters()
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def execute(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.use_extra_node_feature and batch_data.node_feature != None:
            extra_node_feature = self.extra_emb(batch_data.node_feature)
        else:
            extra_node_feature = None
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, batch=batch, r=self.cutoff)
        num_nodes = z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, extra_node_feature, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(jittor.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)

        return u

if __name__ == "__main__":
    # test data
    class Data:
        def __init__(self):
            n = 40
            z = jittor.randint(1, 10, shape=(n,))
            pos = jittor.rand(n, 3)
            batch = jittor.zeros(n)
            batch[20:] = 1
            self.z = z
            self.pos = pos
            self.batch = batch
    data = Data()
    model = SphereNet()
    result = model(data)
    print(result)
