import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.attention import MultiheadAttention
from jittor_geometric.nn.dense.time_encoder import TimeEncoder


class DyGFormer(nn.Module):
    r"""The implementation of DyGFormer model from the
    `"Towards Better Dynamic Graph Learning: New Architecture and Unified Library"
    <https://arxiv.org/abs/2303.13047>`_ paper. Most of the code is adapted from the original implementation in https://github.com/yule-BUAA/DyGLib/.

    .. note::

        For an example of using DyGFormer, see `examples/dygformer_example.py`.

    Args:
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param bipartite: bool, whether to use bipartite graph. This is implemented by Jittor-Geometric Team to accelerate the computation of neighbor co-occurrence features for bipartite graphs. 
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, bipartite: bool = False):
        super(DyGFormer, self).__init__()

        self.node_raw_features = jt.nn.Parameter(
            jt.Var(node_raw_features), requires_grad=False)
        self.edge_raw_features = jt.nn.Parameter(
            jt.Var(edge_raw_features), requires_grad=False)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.bipartite_graph = bipartite

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, bipartite=self.bipartite_graph)

        self.projection_layer_node = nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        self.projection_layer_edge = nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        self.projection_layer_time = nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        self.projection_layer_neighbor_co_occurrence = nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)

        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim,
                               num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # modify: pad the seq with the same length for source and destination nodes
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(src_nodes_neighbor_ids_list)):
            assert len(src_nodes_neighbor_ids_list[idx]) == len(
                src_nodes_edge_ids_list[idx]) == len(src_nodes_neighbor_times_list[idx])
            if len(src_nodes_neighbor_ids_list[idx]) > self.max_input_sequence_length - 1:
                # cut the sequence by taking the most recent self.max_input_sequence_length interactions
                src_nodes_neighbor_ids_list[idx] = src_nodes_neighbor_ids_list[idx][-(
                    self.max_input_sequence_length - 1):]
                src_nodes_edge_ids_list[idx] = src_nodes_edge_ids_list[idx][-(
                    self.max_input_sequence_length - 1):]
                src_nodes_neighbor_times_list[idx] = src_nodes_neighbor_times_list[idx][-(
                    self.max_input_sequence_length - 1):]
            if len(src_nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(src_nodes_neighbor_ids_list[idx])
        for idx in range(len(dst_nodes_neighbor_ids_list)):
            assert len(dst_nodes_neighbor_ids_list[idx]) == len(
                dst_nodes_edge_ids_list[idx]) == len(dst_nodes_neighbor_times_list[idx])
            if len(dst_nodes_neighbor_ids_list[idx]) > self.max_input_sequence_length - 1:
                # cut the sequence by taking the most recent self.max_input_sequence_length interactions
                dst_nodes_neighbor_ids_list[idx] = dst_nodes_neighbor_ids_list[idx][-(
                    self.max_input_sequence_length - 1):]
                dst_nodes_edge_ids_list[idx] = dst_nodes_edge_ids_list[idx][-(
                    self.max_input_sequence_length - 1):]
                dst_nodes_neighbor_times_list[idx] = dst_nodes_neighbor_times_list[idx][-(
                    self.max_input_sequence_length - 1):]
            if len(dst_nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(dst_nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % self.patch_size != 0:
            max_seq_length += (self.patch_size -
                               max_seq_length % self.patch_size)
        assert max_seq_length % self.patch_size == 0
        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_seq_length=max_seq_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_seq_length=max_seq_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Var, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Var, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Var, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Var, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Var, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Var, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Var, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Var, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Var, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Var, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Var, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Var, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Var, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Var, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
        # Var, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer_node(
            src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer_edge(
            src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer_time(
            src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer_neighbor_co_occurrence(
            src_patches_nodes_neighbor_co_occurrence_features)

        # Var, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer_node(
            dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer_edge(
            dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer_time(
            dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer_neighbor_co_occurrence(
            dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Var, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = jt.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = jt.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = jt.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = jt.cat(
            [src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        # Var, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = jt.stack(patches_data, dim=2)
        # Var, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(
            batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # Var, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        # src_patches_data, Var, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Var, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:,
                                        src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Var, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = jt.mean(src_patches_data, dim=1)
        # dst_patches_data, Var, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = jt.mean(dst_patches_data, dim=1)

        # Var, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Var, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, max_seq_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros(
            (len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros(
            (len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros(
            (len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(
                    nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(
                    nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(
                    nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Var, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[jt.array(
            padded_nodes_neighbor_ids)]
        # Var, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[jt.array(
            padded_nodes_edge_ids)]
        # Var, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=jt.array(
            node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float())

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[jt.array(
            padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: jt.Var, padded_nodes_edge_raw_features: jt.Var,
                    padded_nodes_neighbor_time_features: jt.Var, padded_nodes_neighbor_co_occurrence_features: jt.Var = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Var, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Var, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Var, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Var, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Vars with shape (num_patches, ), each Var with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
            patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Var, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = jt.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.node_feat_dim)
        # Var, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = jt.stack(patches_nodes_edge_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Var, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = jt.stack(patches_nodes_neighbor_time_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = jt.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

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


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int,bipartite=False):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1,
                      out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))
        self.bipartite_graph = bipartite

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        if self.bipartite_graph:
            src_padded_nodes_neighbor_ids = jt.array(src_padded_nodes_neighbor_ids)
            dst_padded_nodes_neighbor_ids = jt.array(dst_padded_nodes_neighbor_ids)
            src = src_padded_nodes_neighbor_ids[:, 0].unsqueeze(-1)
            dst = dst_padded_nodes_neighbor_ids[:, 0].unsqueeze(-1)
            dst_in_src = (src_padded_nodes_neighbor_ids == dst).sum(-1).int()
            src_in_dst = (dst_padded_nodes_neighbor_ids == src).sum(-1).int()
            src_unique, src_reverse, src_counts = [], [], []
            for src_row in src_padded_nodes_neighbor_ids:
                unique, reverse, counts = jt.unique(
                    src_row, return_inverse=True, return_counts=True)
                src_unique.append(unique)
                src_reverse.append(reverse)
                src_counts.append(counts)
            src_unique_rows = jt.nn.utils.rnn.pad_sequence(
                src_unique, batch_first=True)
            src_inverse_rows = jt.nn.utils.rnn.pad_sequence(
                src_reverse, batch_first=True)
            src_counts_rows = jt.nn.utils.rnn.pad_sequence(
                src_counts, batch_first=True)
            dst_unique, dst_reverse, dst_counts = [], [], []
            for dst_row in dst_padded_nodes_neighbor_ids:
                unique, reverse, counts = jt.unique(
                    dst_row, return_inverse=True, return_counts=True)
                dst_unique.append(unique)
                dst_reverse.append(reverse)
                dst_counts.append(counts)
            dst_unique_rows = jt.nn.utils.rnn.pad_sequence(
                dst_unique, batch_first=True)
            dst_inverse_rows = jt.nn.utils.rnn.pad_sequence(
                dst_reverse, batch_first=True)
            dst_counts_rows = jt.nn.utils.rnn.pad_sequence(
                dst_counts, batch_first=True)
            src_padded_node_neighbor_counts_in_src = src_counts_rows.gather(
                1, src_inverse_rows).float()
            dst_padded_node_neighbor_counts_in_dst = dst_counts_rows.gather(
                1, dst_inverse_rows).float()
            src_padded_node_neighbor_counts_in_src[src_padded_nodes_neighbor_ids == -1] = 0.0
            dst_padded_node_neighbor_counts_in_dst[dst_padded_nodes_neighbor_ids == -1] = 0.0
            src_padded_node_neighbor_counts_in_dst, dst_padded_node_neighbor_counts_in_src = jt.zeros_like(
                src_padded_nodes_neighbor_ids, dtype=jt.float32), jt.zeros_like(dst_padded_nodes_neighbor_ids, dtype=jt.float32)
            # src itself
            src_padded_node_neighbor_counts_in_dst[:, 0] = src_in_dst
            dst_padded_node_neighbor_counts_in_src[:, 0] = dst_in_src
            # count dst in the dst, only one time
            src_padded_node_neighbor_counts_in_dst = jt.where(
                src_padded_nodes_neighbor_ids == dst, 1, src_padded_node_neighbor_counts_in_dst)
            dst_padded_node_neighbor_counts_in_src = jt.where(
                dst_padded_nodes_neighbor_ids == src, 1, dst_padded_node_neighbor_counts_in_src)
            src_padded_nodes_appearances = jt.cat([src_padded_node_neighbor_counts_in_src.unsqueeze(
                -1), src_padded_node_neighbor_counts_in_dst.unsqueeze(-1)], dim=-1)
            dst_padded_nodes_appearances = jt.cat([dst_padded_node_neighbor_counts_in_dst.unsqueeze(
                -1), dst_padded_node_neighbor_counts_in_src.unsqueeze(-1)], dim=-1)
        else:
            src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
            # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
            # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
            for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):

                # src_unique_keys, ndarray, shape (num_src_unique_keys, )
                # src_inverse_indices, ndarray, shape (src_max_seq_length, )
                # src_counts, ndarray, shape (num_src_unique_keys, )
                # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
                src_unique_keys, src_inverse_indices, src_counts = np.unique(
                    src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
                # Var, shape (src_max_seq_length, )
                src_padded_node_neighbor_counts_in_src = src_counts[src_inverse_indices]
                # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
                src_mapping_dict = dict(zip(src_unique_keys, src_counts))

                # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
                # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
                # dst_counts, ndarray, shape (num_dst_unique_keys, )
                # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
                dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                    dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
                # Var, shape (dst_max_seq_length, )
                dst_padded_node_neighbor_counts_in_dst = dst_counts[dst_inverse_indices]
                # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
                dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

                # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
                # Var, shape (src_max_seq_length, )
                func = np.vectorize(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0))
                src_padded_node_neighbor_counts_in_dst = func(src_padded_node_neighbor_ids)
                # src_padded_node_neighbor_counts_in_dst = jt.array(src_padded_node_neighbor_ids.copy(
                # )).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float()
                # Var, shape (src_max_seq_length, 2)
                src_padded_nodes_appearances.append(jt.stack(
                    [src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

                # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
                # Var, shape (dst_max_seq_length, )
                func = np.vectorize(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0))
                dst_padded_node_neighbor_counts_in_src = func(dst_padded_node_neighbor_ids)
                # dst_padded_node_neighbor_counts_in_src = jt.array(dst_padded_node_neighbor_ids.copy(
                # )).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float()
                # Var, shape (dst_max_seq_length, 2)
                dst_padded_nodes_appearances.append(np.stack(
                    [dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], axis=1))

            # Var, shape (batch_size, src_max_seq_length, 2)
            src_padded_nodes_appearances = np.stack(
                src_padded_nodes_appearances, axis=0)
            # Var, shape (batch_size, dst_max_seq_length, 2)
            dst_padded_nodes_appearances = np.stack(
                dst_padded_nodes_appearances, axis=0)

            # set the appearances of the padded node (with zero index) to zeros
            # Var, shape (batch_size, src_max_seq_length, 2)
            src_padded_nodes_appearances[src_padded_nodes_neighbor_ids == 0] = 0.0
            # Var, shape (batch_size, dst_max_seq_length, 2)
            dst_padded_nodes_appearances[dst_padded_nodes_neighbor_ids == 0] = 0.0

        return jt.array(src_padded_nodes_appearances), jt.array(dst_padded_nodes_appearances)

    def execute(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Var, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Var, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Var, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Var, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Var, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Var, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim,
                      out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim,
                      out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def execute(self, inputs: jt.Var):
        """
        encode the inputs by Transformer encoder
        :param inputs: Var, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Var, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Var, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Var, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Var, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Var, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(
            nn.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Var, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs
