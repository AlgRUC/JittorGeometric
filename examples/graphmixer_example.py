import jittor as jt
from jittor import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.datasets.tgb_seq import TGBSeqDataset
from jittor_geometric.loader import TemporalDataLoader
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.tgn import LastNeighborLoader
import jittor.nn as F
from jittor_geometric.nn.models.graphmixer import GraphMixer, MergeLayer
import time
class NeighborSamplerWindow():
    def __init__(self,num_nodes,fix_window_size=20, undirected=True):
        self.fix_window_size=fix_window_size
        self.num_nodes=num_nodes
        self.neighbors_arr=jt.zeros((num_nodes,fix_window_size)).int()
        self.edgeids_arr=jt.zeros((num_nodes,fix_window_size)).int()
        self.times_arr=jt.zeros((num_nodes,fix_window_size))
        self.num_neighbors_arr=jt.zeros((num_nodes)).int()
        self.train_val_neighbors_arr=None
        self.train_val_edgeids_arr=None
        self.train_val_times_arr=None
        self.undirected = undirected

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list=self.neighbors_arr[node_ids]
        nodes_edge_ids_list=self.edgeids_arr[node_ids]
        nodes_neighbor_times_list=self.times_arr[node_ids]

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list
        
    def update_neighbors(self, node_ids: np.ndarray, dst_node_ids: np.ndarray, edge_ids: np.ndarray, node_interact_times: np.ndarray):
        for idx, (node_id, dst_node_id, edge_id, node_interact_time) in enumerate(zip(node_ids,dst_node_ids, edge_ids, node_interact_times)):
            if self.num_neighbors_arr[node_id]==self.fix_window_size:
                self.neighbors_arr[node_id][:-1]=self.neighbors_arr[node_id][1:].clone()
                self.edgeids_arr[node_id][:-1]=self.edgeids_arr[node_id][1:].clone()
                self.times_arr[node_id][:-1]=self.times_arr[node_id][1:].clone()
            self.neighbors_arr[node_id][self.num_neighbors_arr[node_id]]=dst_node_id
            self.edgeids_arr[node_id][self.num_neighbors_arr[node_id]]=edge_id
            self.times_arr[node_id][self.num_neighbors_arr[node_id]]=node_interact_time
            if self.num_neighbors_arr[node_id]<self.fix_window_size-1:
                self.num_neighbors_arr[node_id]+=1
        if self.undirected:
            for idx, (node_id, dst_node_id, edge_id, node_interact_time) in enumerate(zip(dst_node_ids, node_ids, edge_ids, node_interact_times)):
                if self.num_neighbors_arr[node_id]==self.fix_window_size:
                    self.neighbors_arr[node_id][:-1]=self.neighbors_arr[node_id][1:]
                    self.edgeids_arr[node_id][:-1]=self.edgeids_arr[node_id][1:]
                    self.times_arr[node_id][:-1]=self.times_arr[node_id][1:]
                self.neighbors_arr[node_id][self.num_neighbors_arr[node_id]]=dst_node_id
                self.edgeids_arr[node_id][self.num_neighbors_arr[node_id]]=edge_id
                self.times_arr[node_id][self.num_neighbors_arr[node_id]]=node_interact_time
                if self.num_neighbors_arr[node_id]<self.fix_window_size-1:
                    self.num_neighbors_arr[node_id]+=1
    
    def reset(self):
        self.neighbors_arr=jt.zeros((self.num_nodes,self.fix_window_size)).long()
        self.edgeids_arr=jt.zeros((self.num_nodes,self.fix_window_size)).long()
        self.times_arr=jt.zeros((self.num_nodes,self.fix_window_size))
    
    def backup_state(self):
        self.train_val_neighbors_arr=self.neighbors_arr
        self.train_val_edgeids_arr=self.edgeids_arr
        self.train_val_times_arr=self.times_arr
    
    def recover_state(self):
        self.neighbors_arr=self.train_val_neighbors_arr
        self.edgeids_arr=self.train_val_edgeids_arr
        self.times_arr=self.train_val_times_arr

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
    for _, batch_data in enumerate(loader):
        src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
        src_neighbor_list, src_eid_list, src_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.src)
        dst_neighbor_list, dst_eid_list, dst_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.dst)
        neg_dst_neighbor_list, neg_dst_eid_list, neg_dst_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.neg_dst)
        src_emb = model[0].compute_node_temporal_embeddings(src, t, src_neighbor_list, src_eid_list, src_time_list)
        dst_emb = model[0].compute_node_temporal_embeddings(dst, t, dst_neighbor_list, dst_eid_list, dst_time_list)
        neg_dst_emb = model[0].compute_node_temporal_embeddings(neg_dst, t, neg_dst_neighbor_list, neg_dst_eid_list, neg_dst_time_list)
        pos_score = F.sigmoid(model[1](src_emb, dst_emb)).cpu().numpy()
        neg_score = F.sigmoid(model[1](src_emb, neg_dst_emb)).cpu().numpy()
        num_neg = neg_score.shape[0]//pos_score.shape[0]
        if num_neg == 1: # ap
            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score])
            ap = average_precision_score(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            print(f'AP: {ap:.4f}, AUC: {auc:.4f}')
        else: # mrr
            mrr_eval.eval(pos_score, neg_score.view(-1, num_neg))
        return ap    

def train():
    best_ap = 0
    patience = 5
    execute_time, load_time, get_neighbor_time = 0, 0, 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_idx_data_loader_tqdm = tqdm(train_loader, ncols=200)
        load_time_s = time.perf_counter()
        for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):
            src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
            load_time_e = time.perf_counter()
            src_neighbor_list, src_eid_list, src_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.src)
            dst_neighbor_list, dst_eid_list, dst_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.dst)
            neg_dst_neighbor_list, neg_dst_eid_list, neg_dst_time_list = neighbor_loader.get_all_first_hop_neighbors(batch_data.neg_dst)
            get_neighbor_e = time.perf_counter()
            src_emb = model[0].compute_node_temporal_embeddings(src, t, src_neighbor_list, src_eid_list, src_time_list)
            dst_emb = model[0].compute_node_temporal_embeddings(dst, t, dst_neighbor_list, dst_eid_list, dst_time_list)
            neg_dst_emb = model[0].compute_node_temporal_embeddings(neg_dst, t, neg_dst_neighbor_list, neg_dst_eid_list, neg_dst_time_list)
            pos_score = model[1](src_emb, dst_emb)
            neg_score = model[1](src_emb, neg_dst_emb)
            loss=criterion(pos_score, jt.ones_like(pos_score))
            loss+=criterion(neg_score, jt.zeros_like(neg_score))
            optimizer.zero_grad()
            optimizer.step(loss)
            train_losses.append(loss.item())
            load_time += load_time_e - load_time_s
            load_time_s = time.perf_counter()
            execute_time += load_time_s - load_time_e
            get_neighbor_time += get_neighbor_e - load_time_e
            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}, execute time: {execute_time:.4f}s, load time: {load_time:.4f}s, get neighbor time: {get_neighbor_time:.4f}s')
        train_loss = np.mean(train_losses)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        ap = test(val_loader)
        print(f'Epoch: {epoch:03d}, Val AP: {ap:.4f}')
        if ap > best_ap:
            best_ap = ap
        else:
            patience -= 1
            if patience == 0:
                break

node_feat_dims = 100
edge_feat_dims = 100
hidden_dims = 100
num_layers = 2
jt.flags.use_cuda = 1 #jt.has_cuda
num_epochs = 100
time_feat_dim = 100
num_neighbors = 20
dropout = 0.1

criterion = jt.nn.BCEWithLogitsLoss()
data = TGBSeqDataset(root='./data/', name='GoogleLocal')
train_idx=np.nonzero(data.train_mask)[0]
val_idx=np.nonzero(data.val_mask)[0]
test_idx=np.nonzero(data.test_mask)[0]
temporal_data = TemporalData(src=jt.Var(data.src_node_ids.astype(np.int32)), dst=jt.Var(data.dst_node_ids.astype(np.int32)), t=jt.Var(data.time), train_mask=jt.Var(train_idx.astype(np.int32)), val_mask=jt.Var(val_idx.astype(np.int32)), test_mask=jt.Var(test_idx.astype(np.int32)), test_ns=jt.Var(data.test_ns.astype(np.int32)))
train_data, val_data, test_data = temporal_data.train_val_test_split_w_mask()
train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1)
val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)

# Define the neighbor loader
neighbor_loader = NeighborSamplerWindow(data.num_nodes+1)

node_raw_features = np.random.rand(data.num_nodes+1, node_feat_dims)
edge_raw_features = np.random.rand(data.num_edges+1, edge_feat_dims)
dynamic_backbone = GraphMixer(node_raw_features, edge_raw_features, neighbor_loader,time_feat_dim=time_feat_dim, num_tokens=num_neighbors, num_layers=num_layers, dropout=dropout)
link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
model = nn.Sequential(dynamic_backbone, link_predictor)

optimizer = jt.nn.Adam(list(model.parameters()),lr=0.0001)
assoc = jt.empty(data.num_nodes+1, dtype=jt.int32)

train()
test(test_loader)
