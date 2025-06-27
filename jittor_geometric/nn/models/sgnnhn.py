import math
import numpy as np
import jittor as jt
from jittor import nn
from jittor.nn import softmax
from jittor_geometric.nn.conv import MessagePassing


def layer_norm_sum(x):
    ave_x = jt.mean(x, -1).unsqueeze(-1)
    x = x - ave_x
    norm_x = jt.sqrt(jt.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x
    return y

class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = jt.nn.Linear(dim, dim)

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
        reversed_edge_index = jt.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = jt.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = jt.sigmoid(i_r + h_r)
        input_gate = jt.sigmoid(i_i + h_i)
        new_gate = jt.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight=np.random.uniform(-stdv, stdv, weight.shape)

class SGNNHN(nn.Module):
    r"""SGNN-HN applies a star graph neural network to model the complex transition relationship between items in an ongoing session.
        To avoid overfitting, it applies highway networks to adaptively select embeddings from item representations.
    """

    def __init__(self, embedding_size, step, scale, n_items, dropout_seq, max_seq_length):
        super(SGNNHN, self).__init__()
        # load parameters info
        self.embedding_size = embedding_size
        self.step = step
        self.scale = scale
        self.n_items = n_items + 1
        self.dropout_seq= dropout_seq
        self.max_seq_length = max_seq_length
        # item embedding
        self.item_embedding = nn.Embedding(
            int(self.n_items)+1, self.embedding_size, padding_idx=0)
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
            weight=np.random.uniform(-stdv, stdv, weight.shape)

    def att_out(self, hidden, star_node, batch):
        star_node_repeat = jt.index_select(star_node, 0, batch)
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
            star_node_repeat = jt.index_select(star_node, 0, batch)
            sim = (hidden * star_node_repeat).sum(dim=-1,
                                                  keepdim=True) / math.sqrt(self.embedding_size)
            alpha = jt.sigmoid(sim)
            hidden = (1 - alpha) * hidden + alpha * star_node_repeat
            star_node = self.att_out(hidden, star_node, batch)

        seq_hidden = hidden[alias_inputs]
        bs, item_num, _ = seq_hidden.shape
        pos_emb = self.pos_embedding.weight[:item_num]
        pos_emb = pos_emb.unsqueeze(0).expand(bs, -1, -1)
        lenmask = jt.arange(item_num).expand(bs, item_num) < item_seq_len.unsqueeze(1)
        pos_emb=pos_emb*lenmask.unsqueeze(-1)
        seq_hidden = seq_hidden + pos_emb

        # fetch the last hidden state of last timestamp
        item_seq_len[item_seq_len == 0] = 1  # todo: avoid zero division
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)
        q3 = self.linear_three(star_node).view(
            star_node.shape[0], 1, star_node.shape[1])

        alpha = self.linear_four(jt.sigmoid(q1 + q2 + q3))
        a = jt.sum(alpha * seq_hidden *
                      mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(jt.cat([a, ht], dim=1))
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
        graph_batch = self.gnn_transform(jt.arange(len(seq_len)), graph_objs)
        item_seq_len= item_seq_len
        test_item = test_item
        alias_inputs = graph_batch['alias_inputs']
        x = graph_batch['x']
        edge_index = graph_batch['edge_index']
        batch = graph_batch['batch']
        seq_output = self.forward(
            x, edge_index, batch, alias_inputs, item_seq_len)
        seq_output = nn.dropout(seq_output, self.dropout_seq,
                            training=self.training)
        return seq_output, layer_norm_sum(self.item_embedding(test_item.long()))
    
    def predict(self, batch_data):
        seq, seq_len, test_item = batch_data
        seq_len[seq_len == 0] = 1
        graph_objs, item_seq_len, test_item = self.sessionGraphGen(seq, seq_len, test_item)
        graph_batch = self.gnn_transform(jt.arange(len(seq_len)), graph_objs)
        item_seq_len= item_seq_len
        test_item = test_item
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
        for i, seq in enumerate(list(jt.chunk(item_seq, item_seq.shape[0]))):
            seq = seq-self.item_min_idx+1
            seq[seq < 0] = 0
            seq, idx = jt.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = jt.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)
        test_items = test_items - self.item_min_idx + 1
        graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs,
        }
        return graph_objs, item_seq_len, test_items
  
    def gnn_transform(self, index, graph_objs):
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []
        tot_node_num = jt.ones([1], dtype=jt.int64)
        for i in range(index.shape[0]):
            if 'edge_index' in graph_batch:
                graph_batch['edge_index'][i] = graph_batch['edge_index'][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(
                jt.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [jt.zeros(
                [1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]
        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(
                    graph_batch[k], batch_first=True)
            else:
                graph_batch[k] = jt.cat(graph_batch[k], dim=-1)

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

def pad_sequence(sequences, batch_first=True):
    max_len = max([s.shape[0] for s in sequences])
    batch_size = len(sequences)
    out_seqs = jt.zeros(batch_size, max_len, sequences[0].shape[1])
    for i, seq in enumerate(sequences):
        out_seqs[i, :seq.shape[0], :] = seq
    return out_seqs

def global_mean_pool(x, batch):
    dim = -1 if isinstance(x, jt.Var) and x.dim() == 1 else -2
    return jt.scatter(x, batch, dim=dim, reduce='mean')

def global_add_pool(x, batch):
    dim = -1 if isinstance(x, jt.Var) and x.dim() == 1 else -2
    return jt.scatter(x, batch, dim=dim, reduce='sum')
