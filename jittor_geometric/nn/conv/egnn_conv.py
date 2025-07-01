import jittor as jt
from jittor import nn
from jittor import Var
import jittor_geometric
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.typing import Adj, OptVar
from einops import rearrange
import numpy as np
from ..inits import xavier_normal, zeros
from jittor_geometric.utils import degree, scatter
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
            self.weight = nn.Parameter(jt.zeros(in_channels))
            self.bias = nn.Parameter(jt.zeros(in_channels))
        else:
            self.weight = None
            self.bias = None

    def execute(self, x: jt.Var, batch: jt.Var = None, batch_size: int = None) -> jt.Var:
        if self.mode == 'graph':
            # graph-level normalization: calculating the mean and std of the whole graph
            if batch is None:
                mean = jt.mean(x, dim=0, keepdims=True)
                std = jt.std(x, dim=0, unbiased=False, keepdims=True)
            else:
                # normalization based on batch
                if batch_size is None:
                    batch_size = batch.max() + 1
                norm = degree(batch, int(batch_size), dtype=x.dtype).clamp_(min_v=1)
                norm = norm.mul_(x.size(-1)).view(-1, 1)
                mean = scatter(x, batch, dim=0, dim_size=int(batch_size),
                            reduce='sum').sum(dim=-1, keepdim=True) / norm
                x = x - mean.index_select(0, batch)
                var = scatter(x * x, batch, dim=0, dim_size=int(batch_size),
                            reduce='sum').sum(dim=-1, keepdim=True)
                var = var / norm

            normed_x = x / (var + self.eps).sqrt().index_select(0, batch)
            
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
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

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


    def propagate(self, edge_index: Adj, size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max = clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp( jt.cat([hidden_feats, m_i], dim = -1) )
            hidden_out = kwargs["x"] + hidden_out
        else: 
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

