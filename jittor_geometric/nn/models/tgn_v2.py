import copy
from typing import Callable, Dict, Tuple,  Optional

import jittor as jt
from jittor import nn
from jittor.nn import GRUCell, Linear

from jittor_geometric.nn.inits import zeros, glorot
import time
import numpy as np

TGNMessageStoreType = Dict[int, Tuple[jt.Var, jt.Var, jt.Var, jt.Var]]

class TGNMemory_v2(nn.Module):
    r"""Version 2: Optimization of the storage structure of node neighborhood; Acceleration of the 
    aggregation and scatter_max operators.
    The implementation of TGN (Temporal Graph Networks) Memory, as described in the paper
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_.

    TGN is a model designed for learning node representations in temporal graphs. It stores and aggregates 
    messages over time to compute dynamic node embeddings that evolve with graph structure and time.

    .. note::

        For an example of using TGNMemory_v2, see `examples/tgn_example.py`.

    Args:
        :param num_nodes: int, the total number of nodes in the graph
        :param raw_msg_dim: int, the dimensionality of the raw message (input feature dimension)
        :param memory_dim: int, the dimensionality of the memory for each node
        :param time_dim: int, the dimension of the time feature used for temporal embeddings
        :param message_module: Callable, the message passing module that defines how information is passed between nodes
        :param aggregator_module: Callable, the aggregator module that defines how messages are aggregated at each node

    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable, 
                 aggregator_module: Callable):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.memory = jt.empty((num_nodes, memory_dim))
        self.last_update = jt.empty((num_nodes,), dtype=jt.int32)
        self._assoc = jt.empty((num_nodes,), dtype=jt.int32)
        self.memory.requires_grad = False

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self):
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        
        # Manually initialize GRUCell weights and biases
        for param in self.gru.parameters():
            if param.ndim > 1:  # Weight matrix
                jt.init.kaiming_uniform_(param)
            else:  # Bias vector
                jt.init.constant_(param, 0)

        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        self.memory.detach()

    def execute(self, n_id: jt.Var) -> Tuple[jt.Var, jt.Var]:
        if self.is_training():
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: jt.Var, dst: jt.Var, t: jt.Var,
                     raw_msg: jt.Var):
        n_id = jt.concat([src, dst])
        n_id = jt.unique(n_id)

        if self.is_training():
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0, ))
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: jt.Var):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: jt.Var) -> Tuple[jt.Var, jt.Var]:
        self._assoc[n_id] = jt.arange(n_id.shape[0], dtype=jt.int32)

        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
                                                     self.msg_s_module)
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
                                                     self.msg_d_module)
        idx = jt.concat([src_s, src_d], dim=0)
        msg = jt.concat([msg_s, msg_d], dim=0)
        t = jt.concat([t_s, t_d], dim=0)

        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.shape[0])
        memory = self.gru(aggr, self.memory[n_id])
        last_update = jt.scatter(self.last_update, 0, idx, t, reduce='max')[n_id]

        return memory, last_update

    def _update_msg_store(self, src: jt.Var, dst: jt.Var, t: jt.Var,
                          raw_msg: jt.Var, msg_store):
        n_id, perm = src.sort()
        n_id, count = unique_consecutive_jt(n_id)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: jt.Var, msg_store: TGNMessageStoreType,
                     msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = jt.concat(src, dim=0)
        dst = jt.concat(dst, dim=0)
        t = jt.concat(t, dim=0)
        raw_msg = jt.concat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.float32())

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        
        return msg, t, src, dst


    def train(self, mode: bool = True):
        if self.is_training() and not mode:
            self._update_memory(jt.arange(self.num_nodes))
            self._reset_message_store()
        super(TGNMemory_v2, self).train()

def scatter_argmax(src: jt.Var, index: jt.Var, dim: int = 0, dim_size: Optional[int] = None) -> jt.Var:
    assert src.ndim == 1 and index.ndim == 1
    assert dim == 0 or dim == -1
    assert src.numel() == index.numel()

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    res = jt.full((dim_size,), -float('inf'))

    for i in range(index.numel()):
        res[index[i]] = jt.maximum(res[index[i]], src[i])

    out = jt.full((dim_size,), dim_size - 1)

    gathered_res = jt.gather(res, 0, index)
    mask = (src == gathered_res)
    nonzero = jt.nonzero(mask).reshape((-1,))
    
    out[index[nonzero]] = nonzero

    return out

# accelerated scatter_max operator
class ScatterMaxOp(jt.Function):
    def execute(self, src, index, dim_size):
        self.dim_size = dim_size
        
        # use a dictionary to store the maximum value and its position for each index.
        max_dict = {}
        argmax_dict = {}
        
        for i in range(src.shape[0]):
            idx = int(index[i])
            val = float(src[i])
            
            if idx not in max_dict or val > max_dict[idx]:
                max_dict[idx] = val
                argmax_dict[idx] = i
        
        max_values = jt.full((dim_size,), -float('inf'), dtype=src.dtype)
        argmax_indices = jt.full((dim_size,), -1, dtype="int32")
        
        for idx, val in max_dict.items():
            if 0 <= idx < dim_size:
                max_values[idx] = val
                argmax_indices[idx] = argmax_dict[idx]
        
        self.save_vars = max_values, argmax_indices
        return max_values, argmax_indices
    
def scatter_max(src: jt.Var, index: jt.Var, dim_size: int = None):
    if src.numel() == 0 or index.numel() == 0:
        dim_size = dim_size or 0
        return jt.zeros((dim_size,), dtype=src.dtype), jt.full((dim_size,), -1, dtype="int32")
    
    dim_size = dim_size or int(index.max()) + 1
    return ScatterMaxOp().apply(src, index, dim_size)

def unique_consecutive_jt(x: jt.Var) -> Tuple[jt.Var, jt.Var]:
    """
    Pure Jittor implementation of unique_consecutive.
    """
    if x.numel() == 0:
        return jt.array([], dtype=x.dtype), jt.array([], dtype=jt.int32)

    # diff = [False, x[1] != x[0], x[2] != x[1], ..., x[n] != x[n-1]]
    # prepend True to mark the first element always as unique
    diff = jt.concat([jt.array([True]), (x[1:] != x[:-1])])

    unique_vals = x[diff]
    
    # get run lengths: positions of unique -> diff.nonzero()
    idx = jt.nonzero(diff).squeeze(1)  # positions of new segments
    # add len to the end for easy diff
    idx = jt.concat([idx, jt.array([x.shape[0]])])
    counts = idx[1:] - idx[:-1]
    
    return unique_vals, counts

class IdentityMessage(nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def execute(self, z_src: jt.Var, z_dst: jt.Var, raw_msg: jt.Var,
                t_enc: jt.Var):
        return jt.concat([z_src, z_dst, raw_msg, t_enc], dim=-1)

# accelerated aggregation operator
class LastAggregator(nn.Module):
    def execute(self, msg: jt.Var, index: jt.Var, t: jt.Var, dim_size: int):
        _, argmax = scatter_max(t, index, dim_size=dim_size)

        out = jt.zeros((dim_size, msg.shape[-1]), dtype=msg.dtype)
        valid_mask = (argmax >= 0) & (argmax < msg.shape[0])
        
        if jt.any(valid_mask):
            out[valid_mask] = msg[argmax[valid_mask]]

        
        return out

class MeanAggregator(nn.Module):
    def execute(self, msg: jt.Var, index: jt.Var, t: jt.Var, dim_size: int):
        return jt.scatter(jt.zeros((dim_size, msg.shape[-1])), 0, index, msg, reduce='mean')

class TimeEncoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def execute(self, t: jt.Var) -> jt.Var:
        return self.lin(t.view(-1, 1)).cos()

class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int):
        self.size = size

        self.neighbors = jt.empty((num_nodes, size), dtype=jt.int32)
        self.e_id = jt.empty((num_nodes, size), dtype=jt.int32)
        self._assoc = jt.empty((num_nodes,), dtype=jt.int32)

        self.reset_state()

    def __call__(self, n_id: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]
        tmp_n_id = jt.concat([n_id, neighbors])
        n_id = jt.unique(tmp_n_id)
        self._assoc[n_id] = jt.arange(n_id.shape[0], dtype=jt.int32) 
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, jt.stack([neighbors, nodes]), e_id


    def insert(self, src, dst):
        neighbors = jt.cat([src, dst], dim=0)
        nodes = jt.cat([dst, src], dim=0)
        e_id = jt.arange(self.cur_e_id, self.cur_e_id + src.size(0)).repeat(2)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = jt.arange(n_id.numel())

        dense_id = jt.arange(nodes.size(0)) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        e_id = jt.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = jt.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)

        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = jt.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)

# v2: change to use LinkedList, where each node is allocated only the space it needs.
class LinkedListLastNeighborLoader:
    def __init__(self, num_nodes: int, size: int):
        self.size = size
        self.num_nodes = num_nodes
        
        self.neighbor_lists = [[] for _ in range(num_nodes)]  
        self.edge_id_lists = [[] for _ in range(num_nodes)]   
        self.timestamp_lists = [[] for _ in range(num_nodes)] 
        
        self._assoc = jt.full((num_nodes,), -1, dtype=jt.int32)
        
        self.cur_e_id = 0

        self.total_neighbor_count = [0 for _ in range(num_nodes)]
        self.insert_time = 0.0
        self.call_time = 0.0
        self.reset_state()
        
    def insert(self, src: jt.Var, dst: jt.Var):
        start = time.perf_counter()

        src_np = src.numpy()
        dst_np = dst.numpy()
        
        e_ids = np.arange(self.cur_e_id, self.cur_e_id + len(src))
        self.cur_e_id += len(src)
        
        for i in range(len(src)):
            s = int(src_np[i])
            d = int(dst_np[i])
            e_id = e_ids[i]
            
            self._add_neighbor(s, d, e_id)
            self._add_neighbor(d, s, e_id)
        
        end = time.perf_counter()
        self.insert_time += end - start
    
    def _add_neighbor(self, node: int, neighbor: int, e_id: int):
        if len(self.neighbor_lists[node]) >= self.size:
            self.neighbor_lists[node].pop()
            self.edge_id_lists[node].pop()
        
        self.neighbor_lists[node].insert(0, neighbor)
        self.edge_id_lists[node].insert(0, e_id)
    
    def __call__(self, n_id: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
        start = time.perf_counter()
        n_id_np = n_id.numpy()
        
        all_neighbors = []
        all_center_nodes = []
        all_e_ids = []
        
        for node in n_id_np:
            node = int(node)
            neighbors = self.neighbor_lists[node]
            e_ids = self.edge_id_lists[node]
            
            for i in range(len(neighbors)):
                all_center_nodes.append(node)
                all_neighbors.append(neighbors[i])
                all_e_ids.append(e_ids[i])
        
        if not all_center_nodes:
            return n_id, jt.array([], dtype=jt.int32).reshape(2, 0), jt.array([], dtype=jt.int32)
        
        center_nodes_jt = jt.array(all_center_nodes, dtype=jt.int32)
        neighbors_jt = jt.array(all_neighbors, dtype=jt.int32)
        e_ids_jt = jt.array(all_e_ids, dtype=jt.int32)
        
        all_nodes = jt.concat([n_id, neighbors_jt]).unique()
        self._assoc[all_nodes] = jt.arange(all_nodes.shape[0])
        
        edge_index = jt.stack([
            self._assoc[neighbors_jt],  
            self._assoc[center_nodes_jt]  
        ], dim=0)
        end = time.perf_counter()
        self.call_time += end - start
        return all_nodes, edge_index, e_ids_jt
    
    def reset_state(self):
        self.neighbor_lists = [[] for _ in range(self.num_nodes)]
        self.edge_id_lists = [[] for _ in range(self.num_nodes)]
        self.cur_e_id = 0
        self._assoc = jt.full((self.num_nodes,), -1, dtype=jt.int32)

        self.insert_time = 0.0
        self.call_time = 0.0
    
    def memory_usage(self) -> int:
        total_memory = 0
        
        for i in range(self.num_nodes):
            total_memory += len(self.neighbor_lists[i]) * 4  
            total_memory += len(self.edge_id_lists[i]) * 4   
        
        total_memory += self._assoc.size * 4  
        
        return total_memory