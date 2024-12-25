import copy
from typing import Callable, Dict, Tuple,  Optional

import jittor as jt
from jittor import nn
from jittor.nn import RNNCell, Linear

from jittor_geometric.nn.inits import zeros, glorot

DyRepMessageStoreType = Dict[int, Tuple[jt.Var, jt.Var, jt.Var, jt.Var]]
import time


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


def scatter_max(src: jt.Var, index: jt.Var, dim: int = 0, dim_size: Optional[int] = None):
    if src.numel() == 0 or index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        return jt.zeros((dim_size,)), jt.zeros((dim_size,), dtype=index.dtype)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    max_values = jt.full((dim_size,), -float('inf'))
    argmax_indices = jt.full((dim_size,), dim_size - 1, dtype=index.dtype)

    for i in range(index.numel()):
        if src[i] > max_values[index[i]]:
            max_values[index[i]] = src[i]
            argmax_indices[index[i]] = i

    return max_values, argmax_indices


class DyRepMemory(nn.Module):
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
        self.rnn = RNNCell(message_module.out_channels, memory_dim)

        self.register_buffer('memory', jt.empty(num_nodes, memory_dim))
        self.register_buffer('last_update', jt.empty(self.num_nodes, dtype=jt.int32))
        self.register_buffer('_assoc', jt.empty(num_nodes, dtype=jt.int32))

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
        
        # Manually initialize RNNCell weights and biases
        for param in self.rnn.parameters():
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
        # self.memory.stop_grad()
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
        
        memory = self.rnn(aggr, self.memory[n_id])

        last_update = jt.scatter(self.last_update, 0, idx, t, reduce='max')[n_id]
        return memory, last_update

    def _update_msg_store(self, src: jt.Var, dst: jt.Var, t: jt.Var,
                          raw_msg: jt.Var, msg_store):
        n_id, perm = src.sort()
        n_id, count = unique_consecutive(n_id)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: jt.Var, msg_store: DyRepMessageStoreType,
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
        super(DyRepMemory, self).train()

def unique_consecutive(input_tensor):
    # 将 Jittor tensor 转换为 NumPy 数组
    np_array = input_tensor.numpy()
    if np_array.size == 0:
        return jt.array([]), jt.array([])
    
    # 初始化
    unique_elements = []
    counts = []

    # 追踪当前元素和计数
    last_element = np_array[0]
    count = 1

    for element in np_array[1:]:
        if element == last_element:
            count += 1
        else:
            unique_elements.append(last_element)
            counts.append(count)
            last_element = element
            count = 1
    
    # 添加最后一个元素
    unique_elements.append(last_element)
    counts.append(count)

    # 将结果转换回 Jittor tensor
    unique_elements_tensor = jt.array(unique_elements)
    counts_tensor = jt.array(counts)
    
    return unique_elements_tensor, counts_tensor

class IdentityMessage(nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def execute(self, z_src: jt.Var, z_dst: jt.Var, raw_msg: jt.Var,
                t_enc: jt.Var):
        return jt.concat([z_src, z_dst, raw_msg, t_enc], dim=-1)

class LastAggregator(nn.Module):
    def execute(self, msg: jt.Var, index: jt.Var, t: jt.Var, dim_size: int):
        # argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = jt.zeros((dim_size, msg.shape[-1]))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
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
        # Inserts newly encountered interactions into an ever growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = jt.cat([src, dst], dim=0)
        nodes = jt.cat([dst, src], dim=0)
        e_id = jt.arange(self.cur_e_id, self.cur_e_id + src.size(0)).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
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

        # Collect new and old interactions...
        e_id = jt.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = jt.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = jt.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)