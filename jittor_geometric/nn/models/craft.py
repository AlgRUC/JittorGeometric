import jittor as jt
from jittor import nn as nn
from jittor_geometric.nn.dense.MLP import MLP
import copy
import math
from jittor.nn import Embedding
import numpy as np
from jittor_geometric.utils.bprloss import BPRLoss
class CRAFT(jt.nn.Module):

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, initializer_range, n_nodes, max_seq_length, loss_type, use_pos=True, input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=False, num_output_layer=1,  emb_dropout_prob=0.1, skip_connection=False):
        super(CRAFT, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size 
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.emb_dropout_prob = emb_dropout_prob
        self.node_embedding = Embedding(
            int(self.n_nodes)+1, self.hidden_size
        )
        self.use_pos = use_pos
        self.input_cat_time_intervals = input_cat_time_intervals
        if use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        output_dim = 0 
        if self.input_cat_time_intervals:
            trm_input_dim = self.hidden_size * 2
        else:
            trm_input_dim = self.hidden_size
        self.cross_attention = CrossAttention(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=trm_input_dim,
            inner_size=trm_input_dim*4,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        output_dim += trm_input_dim
        if self.output_cat_time_intervals or self.input_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        if self.output_cat_time_intervals:
            output_dim += self.hidden_size
        if self.output_cat_repeat_times:
            output_dim += self.hidden_size
        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim, output_dim=1, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=skip_connection)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)
        self.loss_type = loss_type
        if self.loss_type == "BCE":
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        
    def set_min_idx(self, src_min_idx, dst_min_idx):
        self.src_min_idx = src_min_idx
        self.dst_min_idx = dst_min_idx

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight=jt.array(np.random.normal(loc=0.0, scale=self.initializer_range, size=module.weight.shape))
        elif isinstance(module, nn.LayerNorm):
            module.bias=jt.array(np.zeros(module.bias.shape))
            module.weight=jt.array(np.ones(module.weight.shape))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias=jt.array(np.zeros(module.bias.shape))

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst = None, dst_last_update_times = None):
        bs = src_neighb_seq.shape[0]
        src_neighb_seq_len[src_neighb_seq_len == 0] = 1
        neighb_emb = self.node_embedding(src_neighb_seq)
        if self.output_cat_time_intervals: # for all datasets
            dst_last_update_intervals = cur_times.view(-1,1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times<-1]=-100000 
            dst_last_update_intervals = dst_last_update_intervals
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(dst_last_update_intervals.shape[0], dst_last_update_intervals.shape[1], -1)
            dst_node_time_intervals_feat = self.LayerNorm_time_intervals(dst_node_time_intervals_feat)
            dst_node_time_intervals_feat = self.dropout(dst_node_time_intervals_feat)
        test_dst_emb = self.node_embedding(test_dst)
        test_dst_emb = self.LayerNorm(test_dst_emb.view(bs, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)
        if self.output_cat_repeat_times: # only for seen-dominant datasets
            repeat_times = test_dst.view(bs, test_dst.shape[1], 1) == src_neighb_seq.view(bs, 1, src_neighb_seq.shape[1])
            repeat_times = repeat_times.sum(dim=-1).unsqueeze(-1).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.float()).view(bs, -1, self.hidden_size)
            repeat_times_feat = self.LayerNorm_repeat_times(repeat_times_feat)
            repeat_times_feat = self.dropout(repeat_times_feat)
        if self.use_pos: # default
            position_ids = jt.arange(
                src_neighb_seq.size(1), dtype=jt.int64
            )
            position_ids = position_ids.unsqueeze(0).expand_as(src_neighb_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = neighb_emb + position_embedding
        else:
            input_emb = neighb_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.emb_dropout(input_emb)
        if self.input_cat_time_intervals:# comparison experiments: using time encoding instead of position encoding
            src_neighbor_interact_time_intervals = cur_times.view(-1,1) - neighbors_interact_times
            src_neighbor_interact_time_intervals[src_neighb_seq==0]=-100000 # the neighbor is null
            src_neighb_time_embedding = self.time_projection(src_neighbor_interact_time_intervals.float().view(-1,1)).view(src_neighb_seq.shape[0], src_neighb_seq.shape[1], -1)
            src_neighb_time_embedding = self.LayerNorm_time_intervals(src_neighb_time_embedding)
            src_neighb_time_embedding = self.dropout(src_neighb_time_embedding)
            input_emb = jt.cat([input_emb, src_neighb_time_embedding], dim=-1)
        
        attention_mask = src_neighb_seq != 0
        test_dst_mask = jt.ones(test_dst_emb.shape[0], test_dst_emb.shape[1])
        extended_attention_mask = self.get_attention_mask(test_dst_mask, mask_b=attention_mask)
        output = self.cross_attention(
            test_dst_emb, extended_attention_mask, input_emb, output_all_encoded_layers=True
        )[-1]
        if self.output_cat_time_intervals:
            if output is None:
                output = dst_node_time_intervals_feat
            else:
                output = jt.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            if output is None:
                output = repeat_times_feat
            else:
                output = jt.cat([output, repeat_times_feat], dim=-1).float()
        output = self.output_layer(output.view(-1,output.shape[-1])).view(output.shape[0], output.shape[1], -1)
        return output
    
    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = jt.bmm(mask_a.unsqueeze(1).transpose(1,2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = jt.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]test_dst: [B, 1+num_negs], num_negs=0,1,...
        [3]src_neighb_interact_times: [B, L]
        [4]cur_pred_times: [B]
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]src_neighb_last_update_times: [B, L]
        [7]src_slot_encoding: [B, W], W is the time slot window size
        [8]dst_slot_encoding: [B, W]
        """
        src_neighb_seq = src_neighb_seq - self.dst_min_idx + 1
        test_dst = test_dst - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0
        src_neighb_interact_times = src_neighb_interact_times
        src_neighb_seq_len = src_neighb_seq_len
        logits = self.forward(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst=test_dst, dst_last_update_times=dst_last_update_times)
        if self.loss_type == 'BPR':
            positive_probabilities = logits[:,0].flatten()
            negative_probabilities = logits[:,1:].flatten()
        else:
            positive_probabilities = logits[:,0].sigmoid().flatten()
            negative_probabilities = logits[:,1:].sigmoid().flatten()
        return positive_probabilities, negative_probabilities

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        positive_probabilities, negative_probabilities = self.predict(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times)
        bs = test_dst.shape[0]
        if self.loss_type == 'BPR': 
            negative_probabilities = negative_probabilities.flatten()
            positive_probabilities = positive_probabilities.flatten()
            loss = self.loss_fct(positive_probabilities, negative_probabilities)
        predicts = jt.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = jt.cat([jt.ones(bs), jt.zeros(bs)], dim=0)
        if self.loss_type == 'BCE':
            loss = self.loss_fct(predicts, labels)
        elif self.loss_type != 'BPR':
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented! Only BCE and BPR are supported!")
        return loss, predicts, labels


class MultiHeadCrossAttentionbyHand(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (jt.Var): the input of the multi-head self-attention layer
        attention_mask (jt.Var): the attention mask for input tensor

    Returns:
        hidden_states (jt.Var): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        rotary_emb_type=None,  
        rotary_emb=None,
    ):
        super(MultiHeadCrossAttentionbyHand, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.rotary_emb_type = rotary_emb_type
        self.rotary_emb = rotary_emb
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def execute(self, query, attention_mask, key=None, interaction_time=None):
        # input_tensor: [batch_size, seq_len, hidden_size]
        if key is None:
            key = query
            query_time_slots = interaction_time
            key_time_slots = interaction_time
        else:
            if interaction_time is not None:
                query_time_slots = interaction_time[1]
                key_time_slots = interaction_time[0]
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)

        # query_layer: [batch_size, seq_len, num_heads, head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.rotary_emb is not None:
            if self.rotary_emb_type == 'time':
                query_layer = self.rotary_emb(query_layer, input_pos=query_time_slots)
                key_layer = self.rotary_emb(key_layer, input_pos=key_time_slots)
            else:
                query_layer = self.rotary_emb(query_layer)
                key_layer = self.rotary_emb(key_layer)

        # 重新排列以便矩阵乘法
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 3, 1)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = jt.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = jt.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = hidden_states + query

        return hidden_states

class FeedForward4CrossAttn(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (jt.Var): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (jt.Var): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps, output_dim=None
    ):
        super(FeedForward4CrossAttn, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        if output_dim is not None:
            self.output_dim = output_dim
            self.dense_2 = nn.Linear(inner_size, output_dim)
        else:
            self.output_dim = -1
            self.dense_2 = nn.Linear(inner_size, hidden_size)
            self.dropout = nn.Dropout(hidden_dropout_prob)
    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": nn.relu,
            "swish": self.swish,
            "tanh": jt.tanh,
            "sigmoid": jt.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + jt.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jt.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + jt.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * jt.sigmoid(x)

    def execute(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        if self.output_dim==-1:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states)
            hidden_states = hidden_states + input_tensor

        return hidden_states

class CrossAttentionLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (jt.Var): the input of the multi-head self-attention sublayer
        attention_mask (jt.Var): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (jt.Var): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        rotary_emb_type=None,
        rotary_emb=None,
        output_dim=None,
    ):
        super(CrossAttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadCrossAttentionbyHand(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, rotary_emb_type, rotary_emb
        )
        self.feed_forward = FeedForward4CrossAttn(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            output_dim=output_dim
        )

    def execute(self, query, attention_mask, key=None, interaction_time=None):
        attention_output = self.multi_head_attention(query, attention_mask, key, interaction_time)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class CrossAttention(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        rotary_emb(nn.Module, optional): use RoPE. Default: None

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        rotary_emb=None,
        rotary_emb_type=None,
        output_dim=None,
    ):
        super(CrossAttention, self).__init__()
        layer = CrossAttentionLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            rotary_emb_type=rotary_emb_type,
            rotary_emb=rotary_emb,
            output_dim=output_dim,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def execute(self, query, attention_mask, key=None, output_all_encoded_layers=True, interaction_time=None):
        """
        Args:
            hidden_states (jt.Var): the input of the TransformerEncoder
            attention_mask (jt.Var): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            query = layer_module(query, attention_mask, key, interaction_time)
            if output_all_encoded_layers:
                all_encoder_layers.append(query)
        if not output_all_encoded_layers:
            all_encoder_layers.append(query)
        return all_encoder_layers
    