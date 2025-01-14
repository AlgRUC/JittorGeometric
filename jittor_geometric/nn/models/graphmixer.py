from jittor import Module
import numpy as np 
import jittor as jt
from jittor_geometric.nn.models.tgn import LastNeighborLoader
import jittor.nn as nn
from jittor_geometric.nn.dense.time_encoder import TimeEncoder

class GraphMixer(Module):
    r"""The implementation of GraphMixer model from the
    `"Do We Really Need Complicated Model Architectures For Temporal Networks?"
    <https://arxiv.org/abs/2302.11636>`_ paper. Most of the code is adapted from the DyGLib implementation in https://github.com/yule-BUAA/DyGLib/.

    .. note::

        For an example of using GraphMixer, see `examples/graphmixer_example.py`.

    Args:
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: LastNeighborLoader,
                 time_feat_dim: int, num_tokens: int, num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1):
        super(GraphMixer, self).__init__()

        self.node_raw_features = jt.nn.Parameter(
            jt.Var(node_raw_features), requires_grad=False)
        self.edge_raw_features = jt.nn.Parameter(
            jt.Var(edge_raw_features), requires_grad=False)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout

        self.num_channels = self.edge_feat_dim
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels + self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

        self.neighbor_sampler = neighbor_sampler

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[jt.Var(neighbor_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        nodes_neighbor_time_features = self.time_encoder(timestamps=jt.Var(node_interact_times[:, np.newaxis] - neighbor_times).float())

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[jt.Var(neighbor_node_ids == 0)] = 0.0

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = jt.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)

        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(input_tensor=combined_features)

        # Tensor, shape (batch_size, num_channels)
        combined_features = jt.mean(combined_features, dim=1)

        # node encoder
        # get temporal neighbors of nodes, including neighbor ids
        # time_gap_neighbor_node_ids, ndarray, shape (batch_size, time_gap)
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,node_interact_times=node_interact_times,num_neighbors=time_gap)

        # Tensor, shape (batch_size, time_gap, node_feat_dim)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[jt.Var(time_gap_neighbor_node_ids)]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = jt.Var((time_gap_neighbor_node_ids > 0).astype(np.float32))
        # note that if a node has no valid neighbor (whose valid_time_gap_neighbor_node_ids_mask are all zero), directly set the mask to -np.inf will make the
        # scores after softmax be nan. Therefore, we choose a very large negative number (-1e10) instead of -np.inf to tackle this case
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # Tensor, shape (batch_size, time_gap)
        scores = valid_time_gap_neighbor_node_ids_mask.softmax(dim=1)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = jt.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[jt.Var(node_ids)]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.output_layer(jt.cat([combined_features, output_node_features], dim=1))

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: LastNeighborLoader):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler


class FeedForwardNet(Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        # self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
        #                          nn.GELU(),
        #                          nn.Dropout(dropout),
        #                          nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                #  nn.Dropout(dropout))
        self.linear1 = nn.Linear(input_dim, int(dim_expansion_factor * input_dim))
        self.linear2 = nn.Linear(int(dim_expansion_factor * input_dim), input_dim)

    def execute(self, x: jt.Var):
        """
        feed forward net forward process
        :param x: Var, shape (*, input_dim)
        :return:
        """
        x=self.linear1(x)
        x=nn.gelu(x)
        x=nn.dropout(x, self.dropout, is_train=self.is_training())
        x=self.linear2(x)
        x=nn.dropout(x, self.dropout, is_train=self.is_training())
        
        return x


class MLPMixer(Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def execute(self, input_tensor: jt.Var):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Var, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Var, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Var, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Var, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Var, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Var, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Var, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor

