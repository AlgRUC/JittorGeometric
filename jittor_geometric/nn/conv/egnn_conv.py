import jittor as jt
from jittor import nn
from jittor import Var
import jittor_geometric
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.typing import Adj, OptVar
from einops import rearrange
import numpy as np
from ..inits import xavier_normal, zeros
# helper functions

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** jt.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = jt.cat([x.sin(), x.cos()], dim=-1)
    x = jt.cat((x, orig_x), dim = -1) if include_self else x
    return x

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def execute(self, x):
        return x * jt.sigmoid(x)

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = jt.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale


class LayerNorm(nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-5, affine: bool = True, mode: str = 'graph'):
        super(LayerNorm, self).__init__()
        
        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode
        
        if self.affine:
            self.weight = nn.Parameter(jt.zeros(in_channels))  # 可学习的缩放因子
            self.bias = nn.Parameter(jt.zeros(in_channels))  # 可学习的偏移量
        else:
            self.weight = None
            self.bias = None

    def execute(self, x: jt.Var, batch: jt.Var = None, batch_size: int = None) -> jt.Var:
        if self.mode == 'graph':
            # 图级别的归一化：计算整个图的均值和方差
            if batch is None:
                mean = jt.mean(x, dim=0, keepdims=True)
                std = jt.std(x, dim=0, unbiased=False, keepdims=True)
            else:
                # 按照图的批次归一化
                if batch_size is None:
                    batch_size = batch.max() + 1
                mean = jt.scatter(x, batch, dim=0, reduce='mean', dim_size=batch_size)
                std = jt.scatter(x, batch, dim=0, reduce='std', unbiased=False, dim_size=batch_size)
                
            normed_x = (x - mean) / (std + self.eps).sqrt()

            if self.weight is not None and self.bias is not None:
                normed_x = normed_x * self.weight + self.bias

            return normed_x

        elif self.mode == 'node':
            # 节点级别的归一化：按节点特征归一化
            return nn.functional.layer_norm(x, (self.in_channels,), self.weight, self.bias, self.eps)
        
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, affine={self.affine}, mode={self.mode})'


# global linear attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        # dots = np.einsum('b h i d, b h j d -> b h i j', q.numpy(), k.numpy()) * self.scale
        # Compute the dot product for attention scores: (b, h, n, d) x (b, h, n, d) -> (b, h, n, n)
        dots = jt.matmul(q, k.transpose(0, 1)) * self.scale  # [b, h, n, n]

        if mask is not None:
            mask_value = - jt.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = np.einsum('b h i j, b h j d -> b h i d', attn.numpy(), v.numpy())
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask = mask)
        out     = self.attn2(x, induced)

        x =  out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries


class EGNNConv(MessagePassing):
    """EGNN implementation using Jittor and Jittor Geometric."""

    def __init__(self, feats_dim, pos_dim=3, edge_attr_dim=0, m_dim=16,
                 fourier_features=0, soft_edge=0, norm_feats=False,
                 norm_coors=False, norm_coors_scale_init=1e-2, update_feats=True,
                 update_coors=True, dropout=0., coor_weights_clamp_value = None, aggr='add', **kwargs):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNNConv, self).__init__(**kwargs)

        # Model parameters
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_feats = update_feats
        self.update_coors = update_coors
        self.coor_weights_clamp_value = None

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                        nn.Sigmoid()
        ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1. 
        self.node_norm = LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None


        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            xavier_normal(module.weight)
            zeros(module.bias)

    def execute(self, x: Var, edge_index: Adj, edge_attr: OptVar = None,
                batch: OptVar = None) -> Var:
        """"""
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)

        if edge_attr is not None:
            edge_attr_feats = jt.concat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        # Propagate and compute messages
        hidden_feats, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                 coors=coors, rel_coors=rel_coors, batch=batch)

        return jt.concat([coors_out, hidden_feats], dim=-1)

    def message(self, x_i: Var, x_j: Var, edge_attr: Var) -> Var:
        """Message function."""
        return self.edge_mlp(jt.concat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out: Var, coors_out: Var) -> Var:
        """Update function."""
        if self.update_feats:
            hidden_out = self.node_mlp(jt.concat([aggr_out, coors_out], dim=-1))
        else:
            hidden_out = aggr_out

        if self.update_coors:
            coors_out = self.coors_mlp(aggr_out)

        return hidden_out, coors_out
