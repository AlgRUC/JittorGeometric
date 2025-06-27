import jittor as jt
from jittor import nn
from jittor_geometric.nn.models.transformers import TransformerEncoderbyHand
import numpy as np
class SASRec(jt.nn.Module):

    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, initializer_range, n_items, max_seq_length):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size  # same as embedding_size
        self.inner_size = inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_items = n_items
        self.max_seq_length = max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(
            int(self.n_items+1), int(self.hidden_size), padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoderbyHand(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def set_min_idx(self, user_min_idx, item_min_idx):
        self.user_min_idx = user_min_idx
        self.item_min_idx = item_min_idx

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight=jt.array(np.random.normal(loc=0.0, scale=self.initializer_range, size=module.weight.shape))
        elif isinstance(module, nn.LayerNorm):
            module.bias=jt.array(np.zeros(module.bias.shape))
            module.weight=jt.array(np.ones(module.weight.shape))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias=jt.array(np.zeros(module.bias.shape))

    def forward(self, item_seq, item_seq_len):
        item_seq_len[item_seq_len == 0] = 1
        position_ids = jt.arange(
            item_seq.size(1), dtype=jt.int64
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # jt.bool
        if not bidirectional:
            extended_attention_mask = jt.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = jt.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def calculate_loss(self, batch_data):
        item_seq = batch_data[0] - self.item_min_idx + 1
        item_seq[item_seq < 0] = 0
        item_seq_len = batch_data[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = batch_data[2] - self.item_min_idx + 1
        return seq_output, (self.item_embedding(test_item.long()))
        # if self.loss_type == "BPR":
        #     pass
        # else:  # self.loss_type = 'CE'
        #     test_item_emb = self.item_embedding.weight
        #     logits = jt.matmul(seq_output, test_item_emb.transpose(0, 1))
        #     loss = self.loss_fct(logits, pos_items)
        #     return loss

    def predict(self, interaction):
        item_seq = interaction[0] - self.item_min_idx + 1
        item_seq[item_seq < 0] = 0
        item_seq_len = interaction[1]
        test_item = interaction[2] - self.item_min_idx + 1
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = (seq_output.view(-1, 1, seq_output.shape[-1]) * test_item_emb).sum(dim=-1)
        return scores[:, 0], scores[:, 1:]

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