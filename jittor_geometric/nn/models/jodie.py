import os.path as osp
import sys
import jittor as jt
from jittor.nn import Embedding, Linear, RNNCell
import numpy as np
from jittor import nn

class JODIEEmbedding(jt.nn.Module):
    r"""The implementation of JODIE model for temporal link prediction, as described in the paper
    `"Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks"
    <https://arxiv.org/abs/1908.01207>`_.

    This model learns dynamic embeddings for users and items in temporal graphs by modeling their interactions over time.

    .. note::

        For an example of using JODIEEmbedding, see `examples/jodie_example.py`.

    Args:
        :param embedding_dim: int, the dimensionality of the embedding space for users and items
        :param num_users: int, the total number of users in the graph
        :param num_items: int, the total number of items in the graph
        :param src_mean: float, the mean time shift for source node embeddings
        :param src_std: float, the standard deviation of time shift for source node embeddings
        :param dst_mean: float, the mean time shift for destination node embeddings
        :param dst_std: float, the standard deviation of time shift for destination node embeddings

    """
    def __init__(self, embedding_dim, num_users, num_items, src_mean, src_std, dst_mean, dst_std):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.time_emb = nn.Linear(1, embedding_dim)
        self.user_rnn = nn.RNNCell(embedding_dim * 2, embedding_dim)
        self.item_rnn = nn.RNNCell(embedding_dim * 2, embedding_dim)
        
        # Timestamp buffer (使用register_buffer确保GPU兼容)
        self.register_buffer('last_user_t', jt.zeros(num_users) - 1)
        self.register_buffer('last_item_t', jt.zeros(num_items) - 1)
        
        # Time parameters
        self.src_mean = src_mean
        self.src_std = src_std
        self.dst_mean = dst_mean
        self.dst_std = dst_std

    def execute(self, user_idx, item_idx, timestamp):
        u_emb = self.user_emb(user_idx)
        i_emb = self.item_emb(item_idx)

        delta_t_user = (timestamp - self.last_user_t[user_idx]) / (self.src_std + 1e-9)
        delta_t_item = (timestamp - self.last_item_t[item_idx]) / (self.dst_std + 1e-9)
        
        t_emb_user = self.time_emb(delta_t_user.unsqueeze(-1))
        t_emb_item = self.time_emb(delta_t_item.unsqueeze(-1))
        
        u_input = jt.concat([u_emb, t_emb_user], dim=-1)
        i_input = jt.concat([i_emb, t_emb_item], dim=-1)
        
        updated_u = self.user_rnn(u_input, u_emb)
        updated_i = self.item_rnn(i_input, i_emb)
        
        return updated_u, updated_i

    def update_timestamps(self, user_idx, item_idx, timestamp):
        self.last_user_t.scatter_(0, user_idx, timestamp)
        self.last_item_t.scatter_(0, item_idx, timestamp)

def compute_src_dst_node_time_shifts(src_node_ids, dst_node_ids, node_interact_times):
    src_node_last_timestamps = {}
    dst_node_last_timestamps = {}
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []

    for i in range(len(src_node_ids)):
        src_node_id = src_node_ids[i]
        dst_node_id = dst_node_ids[i]
        node_interact_time = node_interact_times[i]

        if src_node_id not in src_node_last_timestamps:
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps:
            dst_node_last_timestamps[dst_node_id] = 0

        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])

        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time

    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift, dst_node_std_time_shift