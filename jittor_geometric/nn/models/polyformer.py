import jittor as jt
from jittor import nn, Var, Module
import math
import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.special import comb
from scipy.sparse import coo_matrix

from jittor_geometric.utils import add_self_loops, to_undirected, get_laplacian
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.ops import cootocsr, cootocsc, SpmmCsr


class PolyAttn(nn.Module):
    def __init__(self, dataset, args):
        super(PolyAttn, self).__init__()
        self.K = args.K + 1
        self.base = args.base
        self.norm = nn.LayerNorm(args.hidden)
        self.n_head = args.n_head
        self.multi = args.multi
        self.d_head = args.hidden // args.n_head
        
        self.token_wise_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden, int(args.hidden * self.multi)),
                nn.ReLU(),
                nn.Linear(int(args.hidden * self.multi), args.hidden)
            ) for _ in range(self.K)
        ])

        self.W_Q = nn.Linear(args.hidden, self.n_head * self.d_head, bias=False)
        self.W_K = nn.Linear(args.hidden, self.n_head * self.d_head, bias=False)

        self.bias_scale = Var(jt.ones(self.n_head, self.K))
        self.bias = Var([((j+1) ** args.q)**(-1) for j in range(self.K)])

        self.dprate = args.dprate

    def execute(self, src):
        batch_size = src.shape[0]
        origin_src = src
        src = self.norm(src)
        token = src
        value = src

        if len(self.token_wise_network) != self.K:
            raise ValueError(f"token_wise_network length ({len(self.token_wise_network)}) does not match K ({self.K})")
        
        token = jt.stack([layer(token[:, idx, :]) for idx, layer in enumerate(self.token_wise_network)], dim=1)

        query = self.W_Q(token)
        key = self.W_K(token)
        
        q_heads = query.view(query.shape[0], query.shape[1], self.n_head, self.d_head).transpose(1, 2)  # [n, n_head, k, d_head]
        k_heads = key.view(key.shape[0], key.shape[1], self.n_head, self.d_head).transpose(1, 2)
        v_heads = value.view(value.shape[0], value.shape[1], self.n_head, -1).transpose(1, 2)
        
        attention_scores = jt.matmul(q_heads, k_heads.transpose(-2, -1)) / jt.sqrt(jt.Var(self.d_head).float())
        attention_scores = jt.tanh(attention_scores)

        attn_mask = self.bias_scale * self.bias.unsqueeze(0)

        attention_scores = attention_scores * attn_mask.reshape(1, attn_mask.shape[0], 1, attn_mask.shape[1])
        
        attention_scores = nn.dropout(attention_scores, p=self.dprate, is_train=self.is_training)

        context_heads = jt.matmul(attention_scores, v_heads)
        context_sequence = context_heads.transpose(1, 2).contiguous().view(batch_size, self.K, -1)

        src = nn.dropout(context_sequence, p=self.dprate, is_train=self.is_training)
        src = src + origin_src
        return src


class FFNNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FFNNetwork, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(ffn_dim, hidden_dim)

    def execute(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x

class FFN(nn.Module):
    def __init__(self, dataset, args):
        super(FFN, self).__init__()
        self.K = args.K + 1
        self.base = args.base
        self.dropout = args.dprate
        self.ffn_norm = nn.LayerNorm(args.hidden)
        self.ffn_net = FFNNetwork(args.hidden, args.d_ffn)

    def execute(self, src):
        origin_src = src
        src = self.ffn_norm(src)
        src = self.ffn_net(src)
        src = nn.dropout(src, p=self.dropout, is_train=self.is_training)
        src = src + origin_src
        return src


class PolyFormerBlock(nn.Module):
    def __init__(self, dataset, args):
        super(PolyFormerBlock, self).__init__()
        self.K = args.K + 1
        self.base = args.base

        self.attnmodule = PolyAttn(dataset, args)
        self.ffnmodule = FFN(dataset, args)

    def execute(self, src):
        src = self.attnmodule(src)
        src = self.ffnmodule(src)
        return src


class PolyFormerModel(nn.Module):
    def __init__(self, dataset, args):
        super(PolyFormerModel, self).__init__()
        self.dropout = args.dropout
        self.nlayers = args.nlayer
        self.dataset = args.dataset

        self.attn = nn.ModuleList([PolyFormerBlock(dataset, args) for _ in range(self.nlayers)])
        self.K = args.K + 1
        self.base = args.base

        self.lin1 = nn.Linear(args.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, args.hidden)
        self.lin3 = nn.Linear(args.hidden, args.num_classes)

    def execute(self, x):
        input_mat = x    
        input_mat = jt.stack(input_mat, dim=1)  # [N, k, d]
        input_mat = self.lin1(input_mat)  

        for block in self.attn:
            input_mat = block(input_mat)

        x = jt.sum(input_mat, dim=1)  # [N, d]
        x = nn.dropout(x, p=self.dropout, is_train=self.is_training)
        x = self.lin2(x)
        x = nn.relu(x)
        x = nn.dropout(x, p=self.dropout, is_train=self.is_training)
        x = self.lin3(x)
        return x
    
def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def load_base(dataname, base_name, K, x, edge_index, edge_attr=None):
    file_path = f'./bases/{dataname}_{base_name}_{K}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            list_mat = pickle.load(f)
    else:
        list_mat = get_base(base_name, K, x, edge_index, edge_attr)
        if not os.path.exists('./bases'):
            os.makedirs('./bases')  
        with open(file_path, 'wb') as f:
            print(f"Saving base to {file_path}")
            pickle.dump(list_mat, f)
    return list_mat

def get_base(base_name, K, x, edge_index, edge_attr=None):
    if base_name == 'mono':
        list_mat = mono_base(K, x, edge_index, edge_attr)
    elif base_name == 'cheb':
        list_mat = cheb_base(K, x, edge_index, edge_attr)
    elif base_name == 'bern':
        list_mat = bern_base(K, x, edge_index, edge_attr)
    elif base_name == 'opt':
        list_mat = opt_base(K, x, edge_index, edge_attr)
    return list_mat

def mono_base(K, x, edge_index, edge_attr):
    v_num = x.shape[0]
    edge_index, edge_weight = gcn_norm(edge_index, edge_attr, v_num, improved=False, add_self_loops=True)
    with jt.no_grad():
        csc = cootocsc(edge_index, edge_weight, v_num)
        csr = cootocsr(edge_index, edge_weight, v_num)
    list_mat = []
    list_mat.append(x)
    tmp_mat = x
    
    for _ in range(K):
        tmp_mat = SpmmCsr(x=tmp_mat, csr=csr)
        list_mat.append(tmp_mat)
    
    return list_mat

def cheb_base(K, x, edge_index, edge_attr):
    v_num = x.shape[0]
    #L=I-D^(-0.5)AD^(-0.5)
    edge_index, edge_attr = get_laplacian(edge_index, edge_attr, normalization='sym', dtype=x.dtype, num_nodes=v_num)
    #L_tilde=L-I
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=-1.0, num_nodes=v_num)

    with jt.no_grad():
        csc = cootocsc(edge_index, edge_attr, v_num)
        csr = cootocsr(edge_index, edge_attr, v_num)

    tmp_mat = SpmmCsr(x=x, csr=csr)
    list_mat = []
    Tx_0 = x
    list_mat.append(Tx_0)
    Tx_1 = SpmmCsr(x=Tx_0, csr=csr)
    list_mat.append(Tx_1)
    for i in range(2, K + 1):
        Tx_2 = 2 * SpmmCsr(Tx_1, csr) - Tx_0
        list_mat.append(Tx_2)
        Tx_0, Tx_1 = Tx_1, Tx_2

    return list_mat

def bern_base(K, x, edge_index, edge_attr):
    edge_weight = edge_attr
    v_num = x.shape[0]
    edge_index1, norm1 = gcn_norm(edge_index, edge_weight, v_num, improved=False, add_self_loops=True)
    edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2.0, num_nodes=v_num)

    with jt.no_grad():
        csc1 = cootocsc(edge_index, norm1, v_num)
        csr1 = cootocsr(edge_index, norm1, v_num)
        csc2 = cootocsc(edge_index2, norm2, v_num)
        csr2 = cootocsr(edge_index2, norm2, v_num)

    list_mat = []
    tmp = []
    tmp.append(x)
    for i in range(K):
        # x = jt.matmul(Matrix_2I_L, x)
        x = SpmmCsr(x=x, csr=csr2)
        tmp.append(x)
    tmp[K] = (comb(K, 0) / (2**K)) * tmp[K]
    list_mat.append(tmp[K])
    for i in range(K):
        x = tmp[K - i - 1]
        # x = jt.matmul(Matrix_L, x)
        x = SpmmCsr(x=x, csr=csr1)
        for _ in range(i):
            x = SpmmCsr(x=x, csr=csr1)
        list_mat.append((comb(K, i + 1) / (2**K)) * x)
    assert len(list_mat) == K + 1
    return list_mat

def opt_base(K, x, edge_index, edge_attr):
    edge_weight = edge_attr
    node_dim = 0
    list_mat = []
    v_num = x.shape[0]
    _, norm_A = gcn_norm(edge_index, edge_weight, v_num, improved=False, add_self_loops=True)
    with jt.no_grad():
        csc = cootocsc(edge_index, norm_A, v_num)
        csr = cootocsr(edge_index, norm_A, v_num)

    blank_noise = jt.randn_like(x) * 1e-7
    x = x + blank_noise
    last_h = x / jt.clamp((jt.norm(x, dim=0)), 1e-8)
    list_mat.append(last_h)
    second_last_h = jt.zeros_like(last_h)
    for _ in range(1, K + 1):
        h_i = SpmmCsr(x=last_h, csr=csr)
        _t = jt.sum(h_i * last_h, dim=0)
        h_i = h_i - _t * last_h
        _t = jt.sum(h_i * second_last_h, dim=0)
        h_i = h_i - _t * second_last_h
        h_i = h_i / jt.clamp((jt.norm(h_i, dim=0)), 1e-8)
        list_mat.append(h_i)
        second_last_h = last_h
        last_h = h_i
    return list_mat



def get_data_load(args, dataset):
    data = dataset[0]
    if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
        args.num_classes = 1
    else:
        args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
    data.list_mat = load_base(args.dataset.lower(), args.base, args.K, data.x, data.edge_index, data.edge_attr)
    return dataset, data







