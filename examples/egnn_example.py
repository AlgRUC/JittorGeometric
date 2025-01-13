from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')  # type: ignore
import jittor as jt
import os.path as osp
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor import nn
from jittor_geometric.nn import EGNNConv, global_add_pool
from jittor_geometric.nn.conv.egnn_conv import GlobalLinearAttention, SiLU
from jittor_geometric.typing import Var
from jittor_geometric.datasets import QM9
import jittor_geometric.transforms as T
from jittor_geometric.jitgeo_loader import DataLoader
import jittor_geometric.jitgeo_loader
from tqdm import tqdm
import numpy as np

# helper functions

def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = jt.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
    return x


class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNNConv
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """
    def __init__(self, n_layers, feats_dim, 
                 pos_dim = 3,
                 edge_attr_dim = 0, 
                 m_dim = 16,
                 out_dim = 19,
                 fourier_features = 0, 
                 soft_edge = 0,
                 embedding_nums=[], 
                 embedding_dims=[],
                 edge_embedding_nums=[], 
                 edge_embedding_dims=[],
                 update_coors=True, 
                 update_feats=True, 
                 norm_feats=True, 
                 norm_coors=False,
                 norm_coors_scale_init = 1e-2, 
                 dropout=0.,
                 coor_weights_clamp_value=None, 
                 aggr="add",
                 global_linear_attn_every = 0,
                 global_linear_attn_heads = 8,
                 global_linear_attn_dim_head = 64,
                 num_global_tokens = 4,
                 recalc=0 ,):
        super().__init__()

        self.n_layers         = n_layers 

        # Embeddings solve here
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers      = nn.ModuleList()
        self.feats_dim        = feats_dim
        self.pos_dim          = pos_dim
        self.edge_attr_dim    = edge_attr_dim
        self.m_dim            = m_dim
        self.out_dim          = out_dim
        self.fourier_features = fourier_features
        self.soft_edge        = soft_edge
        self.norm_feats       = norm_feats
        self.norm_coors       = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats     = update_feats
        self.update_coors     = update_coors
        self.dropout          = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc           = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(jt.randn(num_global_tokens, m_dim))
        
        # instantiate layers
        for i in range(n_layers):
            layer = EGNNConv(feats_dim = feats_dim,
                                pos_dim = pos_dim,
                                edge_attr_dim = edge_attr_dim,
                                m_dim = m_dim,
                                fourier_features = fourier_features, 
                                soft_edge = soft_edge, 
                                norm_feats = norm_feats,
                                norm_coors = norm_coors,
                                norm_coors_scale_init = norm_coors_scale_init, 
                                update_feats = update_feats,
                                update_coors = update_coors, 
                                dropout = dropout, 
                                coor_weights_clamp_value = coor_weights_clamp_value)

            # global attention case
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention(dim = self.feats_dim, 
                                                   heads = global_linear_attn_heads, 
                                                   dim_head = global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            # normal case
            else: 
                self.mpnn_layers.append(layer)
        self.dropout_module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_mlp = nn.Sequential(
            nn.Linear(feats_dim, feats_dim * 2),
            self.dropout_module,
            SiLU(),
            nn.Linear(feats_dim * 2, self.out_dim),
        )
        self.pool = global_add_pool
            

    def execute(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        # regulates wether to embedd edges each layer
        edges_need_embedding = True  
        for i,layer in enumerate(self.mpnn_layers):
            
            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            # attn tokens
            global_tokens = None
            if self.global_tokens is not None:
                unique, amounts = jt.unique(batch, return_counts=True)
                num_idxs = jt.cat([jt.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            # pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch)
            else: 
                # only pass feats to the attn layer
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                # merge attn-ed feats and coords
                x = jt.cat( (x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = true

        x_pos = x[:, :self.pos_dim]
        x_node = self.out_mlp(x[:, self.pos_dim:])
        x_graph = self.pool(x_node, batch)
        
        return x_graph, x_node, x_pos

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))


# sample synthetic data (e.g., random graph)
def generate_data(num_nodes, num_edges):
    x = jt.randn((num_nodes, 6))  # 3 coordinates + 3 features
    edge_index = jt.randint(0, num_nodes, (2, num_edges))  # Random edge indices
    edge_attr = jt.randn((num_edges, 3))  # Random edge attributes
    return x, edge_index, edge_attr


# Define MAE loss function
def mae_loss(pred: Var, target: Var) -> Var:
    return jt.abs(pred - target).mean()


# Run training
def train(model, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch.x = jt.concat([batch.pos, batch.x], dim=-1)
        pred, _, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = mae_loss(pred, batch.y)
        optimizer.step(loss)
        loss_accum += loss

    return float(loss_accum / (step + 1))


def eval(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch.x = jt.concat([batch.pos, batch.x], dim=-1)
        pred, _, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        y_true.append(batch.y)
        y_pred.append(pred)

    y_true = jt.cat(y_true, dim = 0)
    y_pred = jt.cat(y_pred, dim = 0)

    return float(mae_loss(y_pred, y_true))


def main():
    # data
    dataset_name = 'qm9'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/QM9')
    qm9_dataset = QM9(path, transform=T.NormalizeFeatures())
    # random split train/val/test = 8/1/1
    split_dict = qm9_dataset.get_idx_split()

    # dataloader
    train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=8, shuffle=True)
    valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=8, shuffle=False)
    test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=8, shuffle=False)

    # model
    model = EGNN_Sparse_Network(n_layers=3, feats_dim=11, edge_attr_dim=4, m_dim=19, out_dim =19, fourier_features=0, dropout=0.1)
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    best_valid_mae = 1000

    for epoch in range(1, 3):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train_mae = train(model, train_loader, optimizer)

            print('Evaluating...')
            valid_mae = eval(model, valid_loader)

            print('Testing...')
            test_mae = eval(model, test_loader)

            print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
            print(f'Best validation MAE so far: {best_valid_mae}')


if __name__ == "__main__":
    main()