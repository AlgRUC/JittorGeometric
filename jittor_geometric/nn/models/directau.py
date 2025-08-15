import jittor as jt
from jittor import nn, Module
from jittor import init
from jittor import Var

def xavier_uniform_initialization(module, init_linear=True):
    if isinstance(module, nn.Embedding):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Linear) and init_linear:
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

class DirectAU(Module):
    def __init__(self, num_users, num_items, embedding_dim, edge_index, lambda_reg=0.05):
        super().__init__()
        self.edge_index = edge_index

        self.n_users = int(num_users)
        self.n_items = int(num_items)
        self.num_nodes = self.n_users + self.n_items
        self.embedding_dim = int(embedding_dim)
        self.lambda_reg = float(lambda_reg)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_uniform_initialization)
        self.restore_user_embedding = None
        self.restore_item_embedding = None

    def get_ego_embeddings(self):
        user_embs, item_embs = self.user_embedding.weight, self.item_embedding.weight
        return jt.normalize(user_embs, dim=-1), jt.normalize(item_embs, dim=-1)

    @staticmethod
    def alignment_loss(z1, z2, alpha=2):
        return jt.norm(z1 - z2, p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity_loss(z, t=2.0):
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # [N, N, D]
        dist_sq = (diff ** 2).sum(dim=2)  # [N, N]
        mask = 1 - init.eye(z.shape[0])
        dist_sq = dist_sq * mask
        dist_sq = dist_sq[mask == 1]
        return jt.log(jt.exp(-t * dist_sq).mean())

    def directau_loss(self, user_emb, item_emb):
        align_loss = self.alignment_loss(user_emb, item_emb)

        uniform_user_loss = self.uniformity_loss(user_emb)
        uniform_item_loss = self.uniformity_loss(item_emb)
        uniform_loss = self.lambda_reg * (uniform_user_loss + uniform_item_loss) / 2
        return align_loss + uniform_loss

    def execute(self, user_ids, pos_item_ids, neg_item_ids):
        if self.restore_user_embedding is not None or self.restore_item_embedding is not None:
            self.restore_user_embedding = None
            self.restore_item_embedding = None
        user_emb, item_emb = self.get_ego_embeddings()
        pos_user_emb = user_emb[user_ids]
        pos_item_emb = item_emb[pos_item_ids]

        loss = self.directau_loss(pos_user_emb, pos_item_emb)
        return loss

    def full_predict(self, user_ids):
        if self.restore_user_embedding is None or self.restore_item_embedding is None:
            self.restore_user_embedding, self.restore_item_embedding = self.get_ego_embeddings()
        user_vecs = self.restore_user_embedding[user_ids]
        return jt.matmul(user_vecs, self.item_embedding.weight.t())
