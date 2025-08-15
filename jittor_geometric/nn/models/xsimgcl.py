import jittor as jt
from jittor import nn, Module

from jittor import Var
from jittor_geometric.nn.models import LightGCN

class XSimGCL(LightGCN):
    def __init__(self, num_users, num_items, embedding_dim, n_layers, edge_index, reg_weight=1e-5,
                 cl_rate=1e-5, temperature=0.05, eps=0.1, layer_cl=1):
        super().__init__(num_users, num_items, embedding_dim, n_layers, edge_index, reg_weight)
        self.temperature = temperature
        self.cl_rate = cl_rate
        self.eps = eps
        self.layer_cl = layer_cl

    def get_all_embeddings(self, perturbed=False, use_E0=True):
        all_embeddings = self.get_ego_embeddings()
        all_embs_cl = all_embeddings
        if use_E0:
            embeddings_list = [all_embeddings]
            num_layers = self.n_layers + 1
        else:
            embeddings_list = []
            num_layers = self.n_layers
        for layer_idx in range(self.n_layers):
            all_embeddings = self.conv(all_embeddings, self.csc, self.csr)
            if perturbed:
                random_noise = jt.rand_like(all_embeddings)
                all_embeddings = all_embeddings + nn.sign(all_embeddings) * jt.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embeddings)
            if layer_idx == self.layer_cl - 1:
                all_embs_cl = all_embeddings
        lightgcn_all_embeddings = sum(embeddings_list) / num_layers

        user_all_embeddings, item_all_embeddings = jt.split(lightgcn_all_embeddings, [self.n_users, self.n_items], dim=0)
        user_all_embeddings_cl, item_all_embeddings_cl = jt.split(all_embs_cl, [self.n_users, self.n_items], dim=0)
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1 = jt.normalize(x1, dim=-1)
        x2 = jt.normalize(x2, dim=-1)

        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = jt.exp(pos_score / self.temperature)

        ttl_score = jt.matmul(x1, x2.transpose(0, 1))
        ttl_score = jt.exp(ttl_score / self.temperature).sum(dim=1)

        loss = -jt.log(pos_score / ttl_score).sum()
        return loss

    def execute(self, user_ids, item_ids, neg_item_ids):
        if self.restore_user_embedding is not None or self.restore_item_embedding is not None:
            self.restore_user_embedding = None
            self.restore_item_embedding = None
        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl = self.get_all_embeddings(perturbed=True)

        u_embeddings = user_all_embeddings[user_ids]
        pos_embeddings = item_all_embeddings[item_ids]
        neg_embeddings = item_all_embeddings[neg_item_ids]

        # calculate BPR Loss
        pos_scores = (u_embeddings * pos_embeddings).sum(dim=-1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=-1)
        mf_loss = self.loss_fn(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user_ids)
        pos_ego_embeddings = self.item_embedding(item_ids)
        neg_ego_embeddings = self.item_embedding(neg_item_ids)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        unique_user_ids = jt.unique(user_ids)
        unique_item_ids = jt.unique(item_ids)

        # calculate CL Loss
        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[unique_user_ids], user_all_embeddings_cl[unique_user_ids])
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[unique_item_ids], item_all_embeddings_cl[unique_item_ids])

        return mf_loss + self.reg_weight * reg_loss + self.cl_rate * (user_cl_loss + item_cl_loss)
