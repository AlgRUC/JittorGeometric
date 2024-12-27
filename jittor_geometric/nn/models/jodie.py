import jittor as jt
from jittor.nn import Embedding, Linear, RNNCell
import numpy as np

class JODIEEmbedding(jt.nn.Module):
    def __init__(self, embedding_dim, num_users, num_items,
                 src_node_mean_time_shift, src_node_std_time_shift,
                 dst_node_mean_time_shift, dst_node_std_time_shift):
        super(JODIEEmbedding, self).__init__()
        
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)
        
        self.time_embedding = Linear(1, embedding_dim)
        
        self.user_rnn = RNNCell(embedding_dim + embedding_dim, embedding_dim)
        self.item_rnn = RNNCell(embedding_dim + embedding_dim, embedding_dim)
        
        self.last_user_timestamp = jt.ones(num_users) * -1
        self.last_item_timestamp = jt.ones(num_items) * -1
        self.last_user_timestamp.requires_grad = False
        self.last_item_timestamp.requires_grad = False

        
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift = dst_node_mean_time_shift
        self.dst_node_std_time_shift = dst_node_std_time_shift

    def execute(self, user_idx, item_idx, timestamp):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        last_user_time = self.last_user_timestamp[user_idx]
        last_item_time = self.last_item_timestamp[item_idx]
        
        delta_t_user = timestamp - last_user_time
        delta_t_item = timestamp - last_item_time
        
        delta_t_user = (delta_t_user - self.src_node_mean_time_shift) / (self.src_node_std_time_shift + 1e-9)
        delta_t_item = (delta_t_item - self.dst_node_mean_time_shift) / (self.dst_node_std_time_shift + 1e-9)

        delta_t_user = delta_t_user.unsqueeze(-1)
        delta_t_item = delta_t_item.unsqueeze(-1)
        
        time_emb_user = self.time_embedding(delta_t_user)
        time_emb_item = self.time_embedding(delta_t_item)
        
        user_input = jt.concat([user_emb, time_emb_user], dim=-1)
        item_input = jt.concat([item_emb, time_emb_item], dim=-1)
        
        user_emb = self.user_rnn(user_input, user_emb)
        item_emb = self.item_rnn(item_input, item_emb)
        
        self.last_user_timestamp[user_idx] = timestamp
        self.last_item_timestamp[item_idx] = timestamp

        return user_emb, item_emb



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
