import jittor as jt
from jittor import nn, Module

from jittor import Var
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.nn.conv.lightgcn_conv import LightGCNConv
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.ops import SpmmCsr, aggregateWithWeight
from jittor import init
from jittor_geometric.data import Data


def _make_bipartite_edge_index(edge_index, num_users, num_items):
    """(u,i) -> symmetric bipartite edges on [0..U+I-1]."""
    u = edge_index[0]
    v = edge_index[1] + num_users
    edge_src = jt.concat([jt.array(u, dtype='int32'), jt.array(v, dtype='int32')], dim=0)
    edge_dst = jt.concat([jt.array(v, dtype='int32'), jt.array(u, dtype='int32')], dim=0)
    edge_index = jt.stack([edge_src, edge_dst], dim=0)  # [2, 2E]
    num_nodes = num_users + num_items
    return Data(edge_index=edge_index, num_nodes=num_nodes)

def xavier_uniform_initialization(module, init_linear=True):
    if isinstance(module, nn.Embedding):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Linear) and init_linear:
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def execute(self, *embeddings):
        emb_loss = jt.zeros(1)
        for embedding in embeddings:
            emb_loss += jt.norm(embedding.reshape(-1), p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def execute(self, pos_scores, neg_scores):
        loss = -jt.log(jt.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss

class LightGCN(Module):
    def __init__(self, num_users, num_items, embedding_dim, n_layers, edge_index, reg_weight=1e-5):
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_users = int(num_users)
        self.n_items = int(num_items)
        self.n_nodes = self.n_users + self.n_items
        self.embedding_dim = int(embedding_dim)
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        self.loss_fn = BPRLoss()
        self.reg_loss = EmbLoss()

        graph = _make_bipartite_edge_index(edge_index, self.n_users, self.n_items)
        edge_index, edge_weight = gcn_norm(graph.edge_index, num_nodes=self.n_nodes, add_self_loops=False)
        with jt.no_grad():
            self.csc = cootocsc(edge_index, edge_weight, self.n_nodes)
            self.csr = cootocsr(edge_index, edge_weight, self.n_nodes)
        self.conv = LightGCNConv(spmm=True) #

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重
        self._cached_adj_t = None
        self._cached_csc = None
        self.apply(xavier_uniform_initialization)
        self.restore_user_embedding = None
        self.restore_item_embedding = None

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = jt.concat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self, use_E0=True):
        all_embeddings = self.get_ego_embeddings()
        if use_E0:
            embeddings_list = [all_embeddings]
            num_layers = self.n_layers + 1
        else:
            embeddings_list = []
            num_layers = self.n_layers
        for _ in range(self.n_layers):
            all_embeddings = self.conv(all_embeddings, self.csc, self.csr)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = sum(embeddings_list) / num_layers

        user_all_embeddings, item_all_embeddings = jt.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items], dim=0
        )
        return user_all_embeddings, item_all_embeddings

    def execute(self, user_ids, item_ids, neg_item_ids):
        if self.restore_user_embedding is not None or self.restore_item_embedding is not None:
            self.restore_user_embedding = None
            self.restore_item_embedding = None
        user_all, item_all = self.get_all_embeddings()
        user_emb = user_all[user_ids]
        pos_emb = item_all[item_ids]
        neg_emb = item_all[neg_item_ids]
        # 损失计算
        bpr_loss = self.compute_loss(user_emb, pos_emb, neg_emb)
        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user_ids)
        pos_ego_embeddings = self.item_embedding(item_ids)
        neg_ego_embeddings = self.item_embedding(neg_item_ids)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        # print(f'bpr_loss: {bpr_loss}, regloss: {self.reg_weight * reg_loss}')
        loss = bpr_loss + self.reg_weight * reg_loss
        return loss

    def full_predict(self, user_ids):
        if self.restore_user_embedding is None or self.restore_item_embedding is None:
            self.restore_user_embedding, self.restore_item_embedding = self.get_all_embeddings()
        user_all = self.restore_user_embedding
        item_all = self.restore_item_embedding
        user_vecs = user_all[user_ids]
        scores = jt.matmul(user_vecs, item_all.transpose(0, 1))
        return scores.view(-1)

    def compute_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        bpr_loss = self.loss_fn(pos_scores, neg_scores)
        return bpr_loss