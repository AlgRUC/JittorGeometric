import jittor as jt
from jittor import Var, nn, Module
import math
from jittor_geometric.nn import GCNConv 
import numpy as np
import scipy

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.use_bn = use_bn

    def execute(self, data):
        x = data.x
        csc = data.csc
        csr = data.csr
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, csc, csr)
            if self.use_bn:
                x = self.bns[i](x)
            x = nn.relu(x)
            x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.convs[-1](x, csc, csr)
        return x
    

def full_attention_conv(qs, ks, vs, output_attn=False):

    qs = qs / scipy.linalg.norm(qs.numpy()) 
    ks = ks / scipy.linalg.norm(ks.numpy())

    N = qs.shape[0]

    kvs = jt.linalg.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = jt.linalg.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    all_ones = jt.ones([ks.shape[0]])
    ks_sum = jt.linalg.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = jt.linalg.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    attention_normalizer = jt.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += jt.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    if output_attn:
        attention=jt.linalg.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
        attention=attention/normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight


    def execute(self, query_input: Var, source_input: Var, edge_index=None, edge_weight=None, output_attn=False):
        query = self.Wq(query_input).view(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).view(-1, self.num_heads, self.out_channels)

        if self.use_weight:
            value = self.Wv(source_input).view(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.view(-1, 1, self.out_channels)

        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn=True)
        else:
            attention_output = full_attention_conv(query, key, value, output_attn=False)
        final_output = attention_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output

class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList() 
        self.bns.append(nn.LayerNorm(hidden_channels)) 
        for _ in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout_val = dropout
        self.activation = nn.relu
        self.use_bn = use_bn 
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act
  
    def execute(self, data):
        x = data.x
        edge_index, edge_weight = data.edge_index, data.edge_attr
        layer_outputs = []
        x = self.fcs[0](x)

        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = nn.dropout(x, p=self.dropout_val, is_train=self.is_training)
        layer_outputs.append(x)

        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_outputs[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x)

            x = nn.dropout(x, p=self.dropout_val, is_train=self.is_training)
            layer_outputs.append(x)

        return x

class SGFormerModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 tc_num_layers=2, tc_num_heads=1, 
                 tc_alpha=0.5, tc_dropout=0.5, 
                 tc_use_bn=True, tc_use_residual=True, tc_use_weight=True, 
                 tc_use_act=False,
                 use_graph=True, graph_weight=0.8, aggregate='add', gnn_module='gcn'):
        super().__init__()
        self.trans_conv = TransConv(
            in_channels, hidden_channels, num_layers=tc_num_layers, num_heads=tc_num_heads, 
            alpha=tc_alpha, dropout=tc_dropout, use_bn=tc_use_bn, 
            use_residual=tc_use_residual, use_weight=tc_use_weight, use_act=tc_use_act
        )

        self.graph_weight=graph_weight
        self.use_act=tc_use_act

        self.aggregate=aggregate
        if use_graph:
            if gnn_module == 'gcn':
                print("Using GCN for graph convolution")
                self.gnn = GCN(
                    in_channels, hidden_channels, hidden_channels, 
                    tc_num_layers, tc_dropout, tc_use_bn
                )
            else:
                raise ValueError(f"Unsupported GNN module: {gnn_module}")
    
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        if aggregate=='add':
            self.fc=nn.Linear(hidden_channels,out_channels)
        elif aggregate=='cat':
            self.fc=nn.Linear(2*hidden_channels,out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

    def execute(self, data):
        x1 = self.trans_conv(data)
        if self.use_graph and self.gnn is not None:
            x2 = self.gnn(data)
            if self.aggregate == 'add':
                final_x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            elif self.aggregate == 'cat':
                final_x = jt.concat([x1, x2], dim=1)
        else:
            final_x = x1
        
        final_x = self.fc(final_x)
        return final_x


