
import jittor as jt
from jittor import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.datasets.tgb_seq import TGBSeqDataset
from jittor_geometric.loader import TemporalDataLoader
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.graphmixer import GraphMixer, MergeLayer
import time
from jittor_geometric.datasets import JODIEDataset
class NeighborSamplerOrigin():

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        super().__init__()
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)
        if node_interact_times is None: # None means to get all the neighbors
            node_interact_times=np.ones_like(node_ids)*np.finfo(np.float32).max 
        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(np.array(node_ids), np.array(node_interact_times))):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = jt.Var(node_neighbor_sampled_probabilities).float().softmax(dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,node_interact_times=node_interact_times,num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),node_interact_times=nodes_neighbor_times_list[-1].flatten(),num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,interact_time=node_interact_time,return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)

def get_neighbor_sampler(data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src.max(), data.dst.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(np.array(data.src), np.array(data.dst), np.array(data.edge_ids), np.array(data.t)):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSamplerOrigin(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)

class MRR_Evaluator(object):
  def __init__(self) -> None:
    pass

  def eval(self, y_pred_pos, y_pred_neg):
    if jt is not None and isinstance(y_pred_pos, jt.Var):
        y_pred_pos = y_pred_pos.detach().cpu().numpy()
    if jt is not None and isinstance(y_pred_neg, jt.Var):
        y_pred_neg = y_pred_neg.detach().cpu().numpy()
    if not isinstance(y_pred_pos, np.ndarray) or not isinstance(y_pred_neg, np.ndarray):
        raise RuntimeError(
            "Arguments to Evaluator need to be either numpy ndarray or jittor Var!"
        )
    batch_size = y_pred_pos.shape[0]
    y_pred_pos = y_pred_pos.reshape(-1, 1)
    y_pred_neg = y_pred_neg.reshape(batch_size,-1)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1./ranking_list.astype(np.float32)
    return mrr_list
  
def test(loader):
    mrr_eval = MRR_Evaluator()
    model.eval()
    res_list = {}
    ap_list, auc_list, mrr_list = [], [], []
    for _, batch_data in enumerate(loader):
        src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
        load_time_e = time.perf_counter()
        src_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=src, node_interact_times=t,num_neighbors=num_neighbors)
        dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=dst, node_interact_times=t,num_neighbors=num_neighbors)
        neg_dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=neg_dst, node_interact_times=t,num_neighbors=num_neighbors)
        pos_score = jt.sigmoid(model[1](src_node_embeddings, dst_node_embeddings)).cpu().numpy()
        neg_score = jt.sigmoid(model[1](src_node_embeddings, neg_dst_node_embeddings)).cpu().numpy()
        num_neg = neg_score.shape[0]//pos_score.shape[0]
        if num_neg == 1: # ap
            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score])
            ap = average_precision_score(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            # print(f'AP: {ap:.4f}, AUC: {auc:.4f}')
            ap_list.append(ap)
            auc_list.append(auc)
        else: # mrr
            mrr_list.extend(mrr_eval.eval(pos_score, neg_score.view(-1, num_neg)))
    if len(ap_list) > 0:
        res_list['AP'] = np.mean(ap_list)
        res_list['AUC'] = np.mean(auc_list)
    if len(mrr_list) > 0:
        res_list['MRR'] = np.mean(mrr_list)
    return res_list

def train():
    best_ap = 0
    patience = 5
    execute_time, load_time, get_neighbor_time, compute_time = 0, 0, 0, 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_idx_data_loader_tqdm = tqdm(train_loader, ncols=120)
        load_time_s = time.perf_counter()
        for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):
            src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
            load_time_e = time.perf_counter()
            src_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=src, node_interact_times=t,num_neighbors=num_neighbors)
            dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=dst, node_interact_times=t,num_neighbors=num_neighbors)
            neg_dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=neg_dst, node_interact_times=t,num_neighbors=num_neighbors)
            pos_score = model[1](src_node_embeddings, dst_node_embeddings)
            neg_score = model[1](src_node_embeddings, neg_dst_node_embeddings)
            loss=criterion(pos_score, jt.ones_like(pos_score))
            loss+=criterion(neg_score, jt.zeros_like(neg_score))
            optimizer.zero_grad()
            optimizer.step(loss)
            train_losses.append(loss.item())
            load_time += load_time_e - load_time_s
            load_time_s = time.perf_counter()
            execute_time += load_time_s - load_time_e
            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
        train_loss = np.mean(train_losses)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        model[0].set_neighbor_sampler(full_neighbor_sampler) 
        ap = test(val_loader)
        print(f'Epoch: {epoch:03d}, Val: {ap}')
        if ap['AP'] > best_ap:
            best_ap = ap['AP']
        else:
            patience -= 1
            if patience == 0:
                break

node_feat_dims = 172
edge_feat_dims = 172
hidden_dims = 100
num_layers = 2
jt.flags.use_cuda = 1 #jt.has_cuda
num_epochs = 100
time_feat_dim = 100
num_neighbors = 30
dropout = 0.1

criterion = jt.nn.BCEWithLogitsLoss()
dataset_name = 'wikipedia'
path='./data/'
if dataset_name in ['GoogleLocal', 'Yelp', 'Taobao', 'ML-20M' 'Flickr', 'YouTube', 'Patent', 'WikiLink', 'wikipedia',]:
    dataset = TGBSeqDataset(root=path, name='GoogleLocal')
    train_idx=np.nonzero(dataset.train_mask)[0]
    val_idx=np.nonzero(dataset.val_mask)[0]
    test_idx=np.nonzero(dataset.test_mask)[0]
    edge_ids=np.arange(dataset.num_edges)
    temporal_data = TemporalData(src=jt.Var(dataset.src_node_ids.astype(np.int32)), dst=jt.Var(dataset.dst_node_ids.astype(np.int32)), t=jt.Var(dataset.time), train_mask=(train_idx.astype(np.int32)), val_mask=(val_idx.astype(np.int32)), test_mask=(test_idx.astype(np.int32)), test_ns=jt.Var(dataset.test_ns.astype(np.int32)), edge_ids=(edge_ids.astype(np.int32)))
    train_data, val_data, test_data = temporal_data.train_val_test_split_w_mask()
    train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1)
    val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
    test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)
elif dataset_name in [ 'reddit', 'mooc', 'lastfm']:
    dataset = JODIEDataset(path, name=dataset_name) # wikipedia, mooc, reddit, lastfm
    data = dataset[0]
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    data.num_edges = data.num_events
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)

# Define the neighbor loader
train_neighbor_sampler = get_neighbor_sampler(train_data, 'recent',seed=0)
full_neighbor_sampler = get_neighbor_sampler(temporal_data, 'recent',seed=1)

node_raw_features = np.zeros((dataset.num_nodes+1, node_feat_dims))
if isinstance(dataset, JODIEDataset):
    edge_raw_features = data.msg
else:
    edge_raw_features = dataset.edge_feat

if edge_raw_features is not None:
    if edge_raw_features.shape[0] != data.num_edges:
        edge_raw_features = np.concatenate((np.zeros((1, edge_raw_features.shape[1])), edge_raw_features), axis=0)
    edge_feat_dims = edge_raw_features.shape[1]
else:
    edge_raw_features = np.zeros((dataset.num_edges+1, edge_feat_dims))
dynamic_backbone = GraphMixer(node_raw_features, edge_raw_features, train_neighbor_sampler,time_feat_dim=time_feat_dim, num_tokens=num_neighbors, num_layers=num_layers, dropout=dropout)
link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],hidden_dim=node_raw_features.shape[1], output_dim=1)
model = nn.Sequential(dynamic_backbone, link_predictor)

optimizer = jt.nn.Adam(list(model.parameters()),lr=0.0001)

train()
print(test(test_loader))
