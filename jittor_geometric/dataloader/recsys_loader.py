import random
from collections import defaultdict
import jittor as jt

class RecsysDataLoader:
    """
    Args:
        edge_index (jt.Var[int32]): shape [2, E], FIRST ROW = user ids, SECOND ROW = item ids (0-based).
        num_items (int): total #items (IDs in [0, num_items-1]).
        batch_size (int): mini-batch size (edges per batch).
        num_neg (int): negatives per positive (>=1).
        shuffle (bool): shuffle edges each epoch.
        seed (int): base seed; each epoch increments it.
    Yields:
        users (jt.Var[int32])      : [B]
        pos_items (jt.Var[int32])  : [B]
        neg_items (jt.Var[int32])  : [B] if num_neg==1 else [B, num_neg]
    """
    def __init__(self, edge_index, num_items, batch_size=1024,
                 num_neg=1, shuffle=True, seed=42):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index must be [2, E]"
        assert num_neg >= 1

        self.batch_size = int(batch_size)
        self.num_neg    = int(num_neg)
        self.shuffle    = bool(shuffle)
        self.seed_base  = int(seed)
        self.num_items  = int(num_items)

        # Pull to python lists (fast iterate + python RNG), still no NumPy.
        users_j = edge_index[0].int32()
        items_j = edge_index[1].int32()
        self.users = [int(x) for x in users_j.tolist()]
        self.items = [int(x) for x in items_j.tolist()]
        self.n = len(self.users)

        # Per-user positive set
        self.user_pos = defaultdict(set)
        for u, i in zip(self.users, self.items):
            self.user_pos[u].add(i)

        self.epoch = 0

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def _sample_neg_k(self, u, k, rng):
        """Sample k negatives for user u (python ints)."""
        upos = self.user_pos[u]
        negl = []
        while len(negl) < k:
            j = rng.randrange(self.num_items)  # python RNG; or use jt.randint + .item()
            if j not in upos:
                negl.append(j)
        return negl

    def __iter__(self):
        rng = random.Random(self.seed_base + self.epoch)
        self.epoch += 1

        idxs = list(range(self.n))
        if self.shuffle:
            rng.shuffle(idxs)

        for s in range(0, self.n, self.batch_size):
            bid = idxs[s:s+self.batch_size]

            u_list = [self.users[i] for i in bid]
            p_list = [self.items[i] for i in bid]

            if self.num_neg == 1:
                n_list = [self._sample_neg_k(u, 1, rng)[0] for u in u_list]
                n_var = jt.array(n_list, dtype='int32')
            else:
                n_list = [self._sample_neg_k(u, self.num_neg, rng) for u in u_list]
                n_var = jt.array(n_list, dtype='int32')

            yield (
                jt.array(u_list, dtype='int32'),
                jt.array(p_list, dtype='int32'),
                n_var
            )