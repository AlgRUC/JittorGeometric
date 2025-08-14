import jittor as jt
from jittor import nn, Module

from jittor import Var
from jittor import init

from collections import defaultdict
import random
import os



def xavier_uniform_initialization(module, init_linear=True):
    if isinstance(module, nn.Embedding):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Linear) and init_linear:
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

class NetworkEmbeddingModel(Module):
    def __init__(self, dataset, num_nodes, embedding_dim, method="DeepWalk", line_order="second"):
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.dataset = str(dataset)
        self.method = method.lower()
        self.line_order = line_order

        self.embeddings = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)
        if self.method == "line" and self.line_order in ("second", "all"):
            self.context_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim)
        else:
            self.context_embeddings = None
        self.bce = nn.BCEWithLogitsLoss()

        self._adj = None
        self._adj_set = None
        self._edges = None
        self._deg = None

        self._alias_nodes = {}
        self._alias_edges = {}
        self._walk_len = None
        self._walks_per_node = None
        self._p = 1.0
        self._q = 1.0
        self._cache = {}

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_uniform_initialization)

    def set_graph(self, edge_index, num_nodes=None, bidirectional=True):
        """
        edge_index: jt.Var[int32] [2,E], row0=src, row1=dst (0-based).
        If bidirectional=True, build undirected adjacency by adding reverse edges.
        """
        if num_nodes is not None:
            self.num_nodes = int(num_nodes)

        u_list = [int(x) for x in edge_index[0].int32().tolist()]
        v_list = [int(x) for x in edge_index[1].int32().tolist()]
        adj = defaultdict(list)
        edges = []

        for u, v in zip(u_list, v_list):
            adj[u].append(v)
            edges.append((u, v))
            if bidirectional:
                adj[v].append(u)
                edges.append((v, u))

        # unique neighbors & build sets
        adj_set = {}
        for k, nbrs in adj.items():
            s = set(nbrs)
            adj[k] = list(s)
            adj_set[k] = s

        self._adj = dict(adj)
        self._adj_set = adj_set
        self._edges = edges

    def prepare_walk_engine(self, p=1.0, q=1.0, walk_length=10, walks_per_node=1, cache_key=None):
        """
        Build alias tables for node2vec/DeepWalk. For DeepWalk, set p=q=1.
        """
        assert self._adj is not None, "Call set_graph(...) before prepare_walk_engine(...)."

        # DeepWalk shortcut
        if self.method == "deepwalk":
            p, q = 1.0, 1.0

        self._p, self._q = float(p), float(q)
        self._walk_len = int(walk_length)
        self._walks_per_node = int(walks_per_node)

        key = cache_key or f"{self.dataset}_{self.method}_p{p}_q{q}_L{walk_length}_W{walks_per_node}"
        if key in self._cache:
            self._alias_nodes, self._alias_edges = self._cache[key]
            return

        # node-level alias (uniform over neighbors)
        alias_nodes = {}
        for u, nbrs in self._adj.items():
            if not nbrs:
                continue
            probs = [1.0/len(nbrs)] * len(nbrs)
            alias_nodes[u] = self._alias_setup(probs)

        # edge-level alias (second-order bias)
        alias_edges = {}
        for src, nbrs in self._adj.items():
            for dst in nbrs:
                dst_nbrs = self._adj.get(dst, [])
                if not dst_nbrs:
                    continue
                probs = []
                prev_set = self._adj_set.get(src, set())
                for x in dst_nbrs:
                    if x == src:
                        probs.append(1.0 / self._p)
                    elif x in prev_set:
                        probs.append(1.0)
                    else:
                        probs.append(1.0 / self._q)
                s = sum(probs)
                probs = [p/s for p in probs] if s > 0 else [1.0/len(dst_nbrs)]*len(dst_nbrs)
                alias_edges[(src, dst)] = self._alias_setup(probs)

        self._alias_nodes, self._alias_edges = alias_nodes, alias_edges
        self._cache[key] = (alias_nodes, alias_edges)

    def execute(self, pos_i, pos_j, neg_j):
        if self.method in ("deepwalk", "node2vec"):
            src_emb = self.embeddings(pos_i)
            pos_emb = self.embeddings(pos_j)
            neg_emb = self.embeddings(neg_j)
            return self.bce_pairwise(src_emb, pos_emb, neg_emb)

        # LINE
        loss = 0.0
        src_emb = self.embeddings(pos_i)
        if self.line_order in ("first", "all"):
            pos_emb_first = self.embeddings(pos_j)
            neg_emb_first = self.embeddings(neg_j)
            loss_first = self.bce_pairwise(src_emb, pos_emb_first, neg_emb_first)
            loss = loss + loss_first
        if self.line_order in ("second", "all"):
            assert self.context_embeddings is not None, "context_embeddings missing (set line_order to include 'second')."
            pos_ctx = self.context_embeddings(pos_j)
            neg_ctx = self.context_embeddings(neg_j)
            loss_second = self.bce_pairwise(src_emb, pos_ctx, neg_ctx)
            loss = loss + loss_second
        return loss

    def edge_scores(self, edge_index):
        if edge_index is None or int(edge_index.shape[1]) == 0:
            return jt.zeros((0,), dtype="float32")

        us = edge_index[0].int32()
        vs = edge_index[1].int32()

        if self.method == "line" and self.line_order in ("second", "all"):
            src = self.embeddings(us)
            dst = self.context_embeddings(vs)
        else:
            src = self.embeddings(us)
            dst = self.embeddings(vs)

        return (src * dst).sum(dim=-1)

    def bce_pairwise(self, src, pos, neg):
        pos_scores = (src * pos).sum(dim=-1)
        neg_scores = (src * neg).sum(dim=-1)
        scores = jt.concat([pos_scores, neg_scores], dim=0)
        labels = jt.concat([jt.ones_like(pos_scores), jt.zeros_like(neg_scores)], dim=0)
        return self.bce(scores, labels)

    def batch_generator(self,
                        batch_size=1024,
                        window_size=5,  # used only for DeepWalk/node2vec
                        num_neg=1,
                        shuffle=True):
        """
        Yield (src, pos, neg) depending on method:
          - DeepWalk / node2vec: (center, context, negative)
          - LINE:               (u,      v,       negative v)
        Shapes are 1D [B]; you can repeat per sample to make multi-negative externally if needed.
        """
        if self.method in ("deepwalk", "node2vec"):
            assert self._walk_len is not None, "Call prepare_walk_engine(...) first."
            start_nodes = list(range(self.num_nodes))
            if shuffle:
                random.shuffle(start_nodes)
            start_nodes = [u for u in start_nodes for _ in range(self._walks_per_node)]

            walks = self._random_walk_batch(start_nodes)  # [Nw, L]

            pos_i, pos_j, neg_j = self._generate_samples_from_walks(walks, window_size, num_neg)
            for s in range(0, len(pos_i), batch_size):
                idx = slice(s, min(s + batch_size, len(pos_i)))
                yield (jt.array(pos_i[idx], dtype="int32"),
                       jt.array(pos_j[idx], dtype="int32"),
                       jt.array(neg_j[idx], dtype="int32"))
        elif self.method == "line":
            assert self._edges is not None and self._adj_set is not None, "Call set_graph(...) first."
            edge_ids = list(range(len(self._edges)))
            if shuffle:
                random.shuffle(edge_ids)

            buf_u, buf_v, buf_neg = [], [], []
            for eid in edge_ids:
                u, v = self._edges[eid]
                for _ in range(num_neg):
                    neg = self._sample_negative_for_u(u)
                    buf_u.append(u)
                    buf_v.append(v)
                    buf_neg.append(neg)
                if len(buf_u) >= batch_size:
                    yield (jt.array(buf_u[:batch_size], dtype="int32"),
                           jt.array(buf_v[:batch_size], dtype="int32"),
                           jt.array(buf_neg[:batch_size], dtype="int32"))
                    buf_u[:batch_size] = []
                    buf_v[:batch_size] = []
                    buf_neg[:batch_size] = []
            if buf_u:
                yield (jt.array(buf_u, dtype="int32"),
                       jt.array(buf_v, dtype="int32"),
                       jt.array(buf_neg, dtype="int32"))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _random_walk_batch(self, start_nodes):
        """Return jt.Var[int32] of shape [num_valid_walks, L]; drop walks shorter than L."""
        walks = []
        L = self._walk_len
        for s in start_nodes:
            w = [s]
            while len(w) < L:
                cur = w[-1]
                nbrs = self._adj.get(cur, [])
                if not nbrs:
                    break
                if len(w) == 1:
                    # first hop: uniform alias on neighbors
                    if cur in self._alias_nodes:
                        J, q = self._alias_nodes[cur]
                        idx = self._alias_draw(J, q)
                        nxt = nbrs[idx]
                    else:
                        nxt = random.choice(nbrs)
                else:
                    prev = w[-2]
                    Jq = self._alias_edges.get((prev, cur))
                    if Jq is None:
                        if cur in self._alias_nodes:
                            J, q = self._alias_nodes[cur]
                            idx = self._alias_draw(J, q)
                            nxt = self._adj[cur][idx]
                        else:
                            nxt = random.choice(nbrs)
                    else:
                        J, q = Jq
                        idx = self._alias_draw(J, q)
                        nxt = self._adj[cur][idx]
                w.append(nxt)
            if len(w) == L:
                walks.append(w)
        return jt.array(walks, dtype="int32") if walks else jt.array([], dtype="int32").reshape(0, L)

    def _generate_samples_from_walks(self, walks, window_size, num_neg):
        """
        walks: jt.Var[int32], [Nw, L]
        return flat python lists: pos_i, pos_j, neg_j (len = #pairs * num_neg)
        """
        pos_i, pos_j, neg_j = [], [], []
        Nw = int(walks.shape[0])
        L = int(walks.shape[1]) if walks.ndim == 2 else 0
        for wi in range(Nw):
            w = [int(x) for x in walks[wi].tolist()]
            for t, center in enumerate(w):
                left = max(0, t - window_size)
                right = min(L, t + window_size + 1)
                for c in range(left, right):
                    if c == t: continue
                    ctx = w[c]
                    for _ in range(num_neg):
                        neg = self._sample_negative_for_u(center)
                        pos_i.append(center)
                        pos_j.append(ctx)
                        neg_j.append(neg)
        return pos_i, pos_j, neg_j

    # ------------ negative sampling ------------

    def _sample_negative_for_u(self, u):
        """Sample a negative neighbor for u not in its adjacency."""
        upos = self._adj_set.get(u, set())
        while True:
            v = random.randrange(self.num_nodes)
            if v not in upos and v != u:
                return v

    # ------------ alias sampling ------------

    @staticmethod
    def _alias_setup(probs):
        K = len(probs)
        q = [0.0] * K
        J = [0] * K
        small, large = [], []
        for i, p in enumerate(probs):
            q[i] = p * K
            (small if q[i] < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            l = large.pop()
            J[s] = l
            q[l] -= (1.0 - q[s])
            (small if q[l] < 1.0 else large).append(l)
        for i in small + large:
            q[i] = 1.0
            J[i] = i
        return jt.array(J, dtype="int32"), jt.array(q, dtype="float32")

    @staticmethod
    def _alias_draw(J, q):
        K = int(q.shape[0])
        kk = random.randrange(K)
        u = random.random()
        return kk if u < float(q[kk].item()) else int(J[kk].item())

    # ---- register positive edges (train/val/test) and sample negatives ----

    def register_pos_edges(self, *edge_indices, undirected=True):
        """
        Register positive edges into a global set so that negative sampling
        can avoid them. If undirected=True, store edges as (min,max).
        """
        if not hasattr(self, "_global_pos_set"):
            self._global_pos_set = set()
        for eidx in edge_indices:
            if eidx is None or int(eidx.shape[1]) == 0:
                continue
            us = eidx[0].int32().tolist()
            vs = eidx[1].int32().tolist()
            for u, v in zip(us, vs):
                u, v = int(u), int(v)
                if undirected:
                    a, b = (u, v) if u < v else (v, u)
                    self._global_pos_set.add((a, b))
                else:
                    self._global_pos_set.add((u, v))

    def sample_neg_pairs(self, n_samples, seed=0, undirected=True):
        """
        Sample n_samples negative edges (i,j) uniformly by rejection sampling,
        avoiding any edge already in self._global_pos_set. Returns jt.Var [2, n].
        """
        import random
        rng = random.Random(seed)
        pos_set = getattr(self, "_global_pos_set", set())
        out = []
        tried = 0
        limit = n_samples * 50 + 1000
        N = int(self.num_nodes)
        while len(out) < n_samples and tried < limit:
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                tried += 1; continue
            if undirected:
                a, b = (i, j) if i < j else (j, i)
            else:
                a, b = i, j
            if (a, b) in pos_set:
                tried += 1; continue
            out.append((i, j))
        if not out:
            return jt.zeros((2, 0), dtype="int32")
        r = jt.array([p[0] for p in out], dtype="int32")
        c = jt.array([p[1] for p in out], dtype="int32")
        return jt.stack([r, c], dim=0)

    def sample_neg_for_edges(self, pos_edge_index, ratio=1, seed=0, undirected=True):
        """
        For a given positive edge_index [2,E_pos], sample ratio * E_pos negatives.
        """
        Epos = int(pos_edge_index.shape[1]) if pos_edge_index is not None else 0
        n_neg = max(0, ratio * Epos)
        return self.sample_neg_pairs(n_neg, seed=seed, undirected=undirected)