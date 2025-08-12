import jittor as jt
from jittor import nn, Module
import math
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import os

def gelu(x):
    return 0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data = jt.normal(0.0, 0.02 / math.sqrt(n_layers), module.weight.shape)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data = jt.normal(0.0, 0.02, module.weight.shape)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def execute(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = attention_dropout_rate
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def execute(self, q, k, v, attn_bias=None):
        orig_q_size = q.shape
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.shape[0]

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        q = q * self.scale
        x = jt.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = nn.softmax(x, dim=3)
        x = nn.dropout(x, p=self.att_dropout, is_train=self.is_training)
        x = jt.matmul(x, v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = dropout_rate

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = dropout_rate

    def execute(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = nn.dropout(y, p=self.self_attention_dropout, is_train=self.is_training)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = nn.dropout(y, p=self.ffn_dropout, is_train=self.is_training)
        x = x + y
        return x

class NAGphormerModel(nn.Module):
    def __init__(
        self,
        hops, 
        n_class,
        input_dim, 
        pe_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        ffn_dim=64, 
        dropout_rate=0.0,
        attention_dropout_rate=0.1
    ):
        super().__init__()
        self.seq_len = hops+1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.Linear1 = nn.Linear(int(self.hidden_dim/2), self.n_class)
        self.scaling = jt.Var([0.5]).float32()


    def execute(self, batched_data):
        tensor = self.att_embeddings_nope(batched_data)
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        output = self.final_ln(tensor)

        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len-1, 1)
        split_tensor = jt.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(jt.concat([target, neighbor_tensor], dim=2))
        layer_atten = nn.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = jt.sum(neighbor_tensor, dim=1, keepdims=True)
        output = (node_tensor + neighbor_tensor).squeeze(1)  
        output = self.Linear1(nn.relu(self.out_proj(output)))
        return nn.log_softmax(output, dim=1)

def col_normalize(mx):
    """Column-normalize numpy or scipy matrix"""
    scaler = StandardScaler()
    mx = scaler.fit_transform(mx)
    return mx

def nor_matrix(adj, a_matrix):
    nor_matrix = adj * a_matrix
    row_sum = nor_matrix.sum(axis=1, keepdims=True)
    nor_matrix = nor_matrix / row_sum
    return nor_matrix

def normalize_features(mx):
    """Row-normalize sparse matrix or ndarray"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy_batch(output, labels):
    if isinstance(labels, tuple):
        labels = labels[0]
    if hasattr(labels, "ndim") and labels.ndim > 1:
        labels = labels.squeeze()
    labels = labels.int32()
    preds, _ = jt.argmax(output, dim=1)
    correct = (preds == labels).float32().sum()
    return correct

def laplacian_positional_encoding(adj, pos_enc_dim):
    n = adj.shape[0]
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(n) - D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
    try:
        from scipy.sparse.linalg import eigsh
        EigVal, EigVec = eigsh(L, k=pos_enc_dim+1, which='SM')
    except Exception as e:
        EigVal, EigVec = np.linalg.eigh(L.toarray())
        EigVec = EigVec[:, :pos_enc_dim+1]
    lap_pos_enc = EigVec[:, 1:pos_enc_dim+1]
    return lap_pos_enc.astype(np.float32)

def re_features(adj, features, K):
    N, d = features.shape
    nodes_features = np.empty((N, 1, K+1, d), dtype=np.float32)
    nodes_features[:, 0, 0, :] = features
    x = features.copy()
    for i in range(K):
        x = adj.dot(x) if sp.issparse(adj) else np.matmul(adj, x)
        nodes_features[:, 0, i+1, :] = x
    return nodes_features.squeeze()   # (N, K+1, d)


