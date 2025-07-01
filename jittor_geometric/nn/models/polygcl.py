import math
import jittor as jt
import jittor.nn as nn
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.utils import get_laplacian, add_self_loops
from jittor_geometric.utils.gssl_utils import cheby


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, n_classes)

    def execute(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def execute(self, h1, h2, h3, h4, c):
        c_x = c.expand_as(h1).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x).squeeze(1)
        sc_2 = self.fn(h1, c_x).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x).squeeze(1)
        sc_4 = self.fn(h3, c_x).squeeze(1)

        logits = jt.concat((sc_1, sc_2, sc_3, sc_4))

        return logits


def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = jt.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] + h[idx - 1]
    return temp


def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = jt.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] - h[idx - 1]
    return temp


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, node_dim=0):
        super(ChebnetII_prop, self).__init__(aggr='add')

        self.K = K
        self.node_dim = node_dim

        self.initial_val_low = jt.array(2.0).stop_grad()
        self.temp_low = jt.array([2.0 / K for _ in range(K)])
        self.temp_high = jt.array([2.0 / K for _ in range(K)])
        self.initial_val_high = jt.array(0.0).stop_grad()

    def reset_parameters(self):
        self.temp_low.assign(jt.array([2.0 / self.K for _ in range(self.K)]))
        self.temp_high.assign(jt.array([2.0 / self.K for _ in range(self.K)]))

    def execute(self, x, edge_index, edge_weight=None, highpass=True):
        if highpass:
            TEMP = nn.relu(self.temp_high)
            coe_tmp = presum_tensor(TEMP, self.initial_val_high)
        else:
            TEMP = nn.relu(self.temp_low)
            coe_tmp = preminus_tensor(TEMP, self.initial_val_low)

        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # print('edge index', edge_index)
        # print('edge weight', edge_weight)

        # L = I - D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.shape[self.node_dim])

        # L_tilde = L - I
        edge_index_tilde, norm_tilde = add_self_loops(edge_index1, norm1, fill_value=-1.0,
                                                      num_nodes=x.shape[self.node_dim])

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2


        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp_low)


class PolyGCL(nn.Module):
    r"""PolyGCL model from the `"PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters"
    <https://openreview.net/pdf?id=y21ZO6M86t>`_ paper

    Parameters
    -----------
    in_dim: int
        Input feature size.
    out_dim: int
        Output feature size.
    K: int
        Order of polynomial, or maximum number of hops considered for message passing.
    act_fn: nn.Module
        Activation function.
    is_bns: bool
        If set to `True`, uses batch norm in encoder.
    """

    def __init__(self, in_dim, out_dim, K, dprate, dropout, is_bns, act_fn):
        super(PolyGCL, self).__init__()

        self.encoder = ChebNetII(num_features=in_dim, hidden=out_dim, K=K, dprate=dprate, dropout=dropout, is_bns=is_bns, act_fn=act_fn)

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.ReLU()

        self.alpha = jt.array(0.5)
        self.beta = jt.array(0.5)

    def get_embedding(self, edge_index, feat):
        h1 = self.encoder(x=feat, edge_index=edge_index, highpass=True)
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)

        h = jt.multiply(self.alpha, h1) + jt.multiply(self.beta, h2)

        return h.detach()

    def execute(self, edge_index, feat, shuf_feat):
        # positive
        h1 = self.encoder(x=feat, edge_index=edge_index, highpass=True)
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)

        # negative
        h3 = self.encoder(x=shuf_feat, edge_index=edge_index, highpass=True)
        h4 = self.encoder(x=shuf_feat, edge_index=edge_index, highpass=False)

        h = jt.multiply(self.alpha, h1) + jt.multiply(self.beta, h2)

        c = self.act_fn(jt.mean(h, dim=0))

        out = self.disc(h1, h2, h3, h4, c)

        return out


class ChebNetII(nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False, act_fn='relu'):
        super(ChebNetII, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)

        self.prop1 = ChebnetII_prop(K=K)
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def execute(self, x, edge_index, highpass=True):
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, highpass=highpass)
        else:
            x = nn.dropout(x, p=self.dprate, is_train=self.is_training())
            x = self.prop1(x, edge_index, highpass=highpass)

        x = nn.dropout(x, p=self.dropout, is_train=self.is_training())

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x