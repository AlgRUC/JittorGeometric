import math
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax
import torch.nn.functional as F
from models.modules import layer_norm, build_mlp, layer_norm_sum
from torch.nn.utils.rnn import pad_sequence

class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class SGNNHN(nn.Module):
    r"""SGNN-HN applies a star graph neural network to model the complex transition relationship between items in an ongoing session.
        To avoid overfitting, it applies highway networks to adaptively select embeddings from item representations.
    """

    def __init__(self, embedding_size, step, device, scale, n_items, dropout_seq, max_seq_length, loss_type='CE'):
        super(SGNNHN, self).__init__()
        # load parameters info
        self.embedding_size = embedding_size
        self.step = step
        self.device = device
        self.scale = scale
        self.n_items = n_items + 1
        self.dropout_seq= dropout_seq
        self.max_seq_length = max_seq_length
        self.loss_type = loss_type
        print(f"Using loss_type: {self.loss_type}")
        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(
            self.max_seq_length, self.embedding_size)  # position encoding
        self.loss_fct = nn.CrossEntropyLoss()
        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_four = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size)

        # parameters initialization
        self._reset_parameters()

    def set_min_idx(self, user_min_idx, item_min_idx):
        self.user_min_idx = user_min_idx
        self.item_min_idx = item_min_idx

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def att_out(self, hidden, star_node, batch):
        star_node_repeat = torch.index_select(star_node, 0, batch)
        sim = (hidden * star_node_repeat).sum(dim=-1)
        sim = softmax(sim, batch)
        att_hidden = sim.unsqueeze(-1) * hidden
        output = global_add_pool(att_hidden, batch)

        return output

    def forward(self, x, edge_index, batch, alias_inputs, item_seq_len):
        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)
        batch = batch.long()
        star_node = global_mean_pool(hidden, batch)
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)
            star_node_repeat = torch.index_select(star_node, 0, batch)
            sim = (hidden * star_node_repeat).sum(dim=-1,
                                                  keepdim=True) / math.sqrt(self.embedding_size)
            alpha = torch.sigmoid(sim)
            hidden = (1 - alpha) * hidden + alpha * star_node_repeat
            star_node = self.att_out(hidden, star_node, batch)

        seq_hidden = hidden[alias_inputs]
        bs, item_num, _ = seq_hidden.shape
        pos_emb = self.pos_embedding.weight[:item_num]
        pos_emb = pos_emb.unsqueeze(0).expand(bs, -1, -1)
        lenmask = torch.arange(item_num).expand(bs, item_num).to(self.device) < item_seq_len.unsqueeze(1)
        pos_emb=pos_emb*lenmask.unsqueeze(-1)
        seq_hidden = seq_hidden + pos_emb

        # fetch the last hidden state of last timestamp
        item_seq_len[item_seq_len == 0] = 1  # todo: avoid zero division
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)
        q3 = self.linear_three(star_node).view(
            star_node.shape[0], 1, star_node.shape[1])

        alpha = self.linear_four(torch.sigmoid(q1 + q2 + q3))
        a = torch.sum(alpha * seq_hidden *
                      mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return layer_norm_sum(seq_output)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1,
                                         1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def calculate_loss(self, batch_data):
        seq, seq_len, test_item = batch_data
        graph_objs, item_seq_len, test_item = self.sessionGraphGen(seq, seq_len, test_item)
        graph_batch = self.gnn_transform(torch.arange(len(seq_len)), graph_objs, self.device)
        item_seq_len= item_seq_len.to(self.device)
        test_item = test_item.to(self.device)
        alias_inputs = graph_batch['alias_inputs']
        x = graph_batch['x']
        edge_index = graph_batch['edge_index']
        batch = graph_batch['batch']
        seq_output = self.forward(
            x, edge_index, batch, alias_inputs, item_seq_len)
        seq_output = F.dropout(seq_output, self.dropout_seq,
                            training=self.training)
        return seq_output, layer_norm_sum(self.item_embedding(test_item.long()))
        # if self.loss_type == 'BPR':
        #     raise NotImplementedError
        # elif self.loss_type == 'CE':
        #     test_item_emb = layer_norm(self.item_embedding.weight)
        #     logits = torch.matmul(
        #         seq_output, test_item_emb.transpose(0, 1)) * self.scale
        #     loss = self.loss_fct(logits, test_item.long())
        #     return loss
        # elif self.loss_type in ['CE_batch', 'CE_batch_w_inner']: # use the positive items in the batch to be negative items
        #     test_item_emb = layer_norm(self.item_embedding(test_item.long()))
        #     logits = torch.matmul(seq_output, test_item_emb.transpose(0,1)) * self.scale
        #     labels = torch.arange(len(test_item)).to(logits.device)
        #     loss = self.loss_fct(logits, labels)
        #     if self.loss_type == 'CE_batch_w_inner': # add user-user loss and item-item loss
        #         logits = torch.matmul(seq_output, seq_output.transpose(0,1)) * self.scale
        #         loss += self.loss_fct(logits, labels)
        #         logits = torch.matmul(test_item_emb, test_item_emb.transpose(0,1)) * self.scale
        #         loss += self.loss_fct(logits, labels)
        #     return loss

    def predict(self, batch_data):
        seq, seq_len, test_item = batch_data
        seq_len[seq_len == 0] = 1
        graph_objs, item_seq_len, test_item = self.sessionGraphGen(seq, seq_len, test_item)
        graph_batch = self.gnn_transform(torch.arange(len(seq_len)), graph_objs, self.device)
        item_seq_len= item_seq_len.to(self.device)
        test_item = test_item.to(self.device)
        alias_inputs = graph_batch['alias_inputs']
        x = graph_batch['x']
        edge_index = graph_batch['edge_index']
        batch = graph_batch['batch']
        seq_output = self.forward(
            x, edge_index, batch, alias_inputs, item_seq_len)
        test_item_emb = layer_norm_sum(self.item_embedding(test_item))
        scores = (seq_output.view(seq_output.shape[0], -1, seq_output.shape[1]) * test_item_emb).sum(dim=-1) * self.scale
        return scores[:, 0], scores[:, 1:]  # pos_score, neg_score

    def sessionGraphGen(self, item_seq, item_seq_len, test_items=None):
        """
        In session graph, only items are considered. Thus, all item indices need to be reindexed, starting from 1 for convenience.
        """
        x = []
        edge_index = []
        alias_inputs = []
        for i, seq in enumerate(list(torch.chunk(item_seq, item_seq.shape[0]))):
            seq = seq-self.item_min_idx+1
            seq[seq < 0] = 0
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)
        test_items = test_items - self.item_min_idx + 1
        graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs,
        }
        return graph_objs, item_seq_len, test_items
  
    def gnn_transform(self, index, graph_objs, device):
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []
        tot_node_num = torch.ones([1], dtype=torch.long)
        for i in range(index.shape[0]):
            if 'edge_index' in graph_batch:
                graph_batch['edge_index'][i] = graph_batch['edge_index'][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(
                torch.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [torch.zeros(
                [1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]
        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(
                    graph_batch[k], batch_first=True).to(device)
            else:
                graph_batch[k] = torch.cat(graph_batch[k], dim=-1).to(device)

        return graph_batch 

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class SRGNN(nn.Module):

    def __init__(self, embedding_size, step, device, scale, n_items, dropout_seq, max_seq_length, loss_type='CE'):
        super(SRGNN, self).__init__()
        # load parameters info
        self.embedding_size = embedding_size
        self.step = step
        self.device = device
        self.scale = scale
        self.n_items = n_items + 1
        self.dropout_seq= dropout_seq
        self.max_seq_length = max_seq_length
        self.loss_type = loss_type
        print(f"Using loss_type: {self.loss_type}")
        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0)
        self.loss_fct = nn.CrossEntropyLoss()
        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size)

        # parameters initialization
        self._reset_parameters()

    def set_min_idx(self, user_min_idx, item_min_idx):
        self.user_min_idx = user_min_idx
        self.item_min_idx = item_min_idx

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def att_out(self, hidden, star_node, batch):
        star_node_repeat = torch.index_select(star_node, 0, batch)
        sim = (hidden * star_node_repeat).sum(dim=-1)
        sim = softmax(sim, batch)
        att_hidden = sim.unsqueeze(-1) * hidden
        output = global_add_pool(att_hidden, batch)

        return output

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        seq_hidden = hidden[alias_inputs]
        # fetch the last hidden state of last timestamp
        item_seq_len[item_seq_len == 0] = 1 
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return layer_norm(seq_output) 
        # return seq_output

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1,
                                         1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def calculate_loss(self, batch_data):
        seq, seq_len, test_item = batch_data
        graph_objs, item_seq_len, test_item = self.sessionGraphGen(seq, seq_len, test_item)
        graph_batch = self.gnn_transform(torch.arange(len(seq_len)), graph_objs, self.device)
        item_seq_len= item_seq_len.to(self.device)
        test_item = test_item.to(self.device)
        alias_inputs = graph_batch['alias_inputs']
        x = graph_batch['x']
        edge_index = graph_batch['edge_index']
        seq_output = self.forward(
            x, edge_index, alias_inputs, item_seq_len)
        seq_output = F.dropout(seq_output, self.dropout_seq,
                            training=self.training)
        return seq_output, layer_norm(self.item_embedding(test_item.long()))
        # return seq_output, self.item_embedding(test_item.long())

    def predict(self, batch_data):
        seq, seq_len, test_item = batch_data
        graph_objs, item_seq_len, test_item = self.sessionGraphGen(seq, seq_len, test_item)
        graph_batch = self.gnn_transform(torch.arange(len(seq_len)), graph_objs, self.device)
        item_seq_len= item_seq_len.to(self.device)
        test_item = test_item.to(self.device)
        alias_inputs = graph_batch['alias_inputs']
        x = graph_batch['x']
        edge_index = graph_batch['edge_index']
        seq_output = self.forward(
            x, edge_index, alias_inputs, item_seq_len)
        test_item_emb = layer_norm(self.item_embedding(test_item))
        # test_item_emb = self.item_embedding(test_item)
        scores = (seq_output.view(seq_output.shape[0], -1, seq_output.shape[1]) * test_item_emb).sum(dim=-1) * self.scale
        return scores[:, 0], scores[:, 1:]  # pos_score, neg_score

    def sessionGraphGen(self, item_seq, item_seq_len, test_items=None):
        """
        In session graph, only items are considered. Thus, all item indices need to be reindexed, starting from 1 for convenience.
        """
        x = []
        edge_index = []
        alias_inputs = []
        for i, seq in enumerate(list(torch.chunk(item_seq, item_seq.shape[0]))):
            seq = seq-self.item_min_idx+1
            seq[seq < 0] = 0
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)
        test_items = test_items - self.item_min_idx + 1
        graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs,
        }
        return graph_objs, item_seq_len, test_items
  
    def gnn_transform(self, index, graph_objs, device):
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []
        tot_node_num = torch.ones([1], dtype=torch.long)
        for i in range(index.shape[0]):
            if 'edge_index' in graph_batch:
                graph_batch['edge_index'][i] = graph_batch['edge_index'][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(
                torch.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [torch.zeros(
                [1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]
        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(
                    graph_batch[k], batch_first=True).to(device)
            else:
                graph_batch[k] = torch.cat(graph_batch[k], dim=-1).to(device)

        return graph_batch 

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
