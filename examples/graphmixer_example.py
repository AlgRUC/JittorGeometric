import sys
import os.path as osp
import os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
import jittor as jt
from jittor import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.datasets.tgb_seq import TGBSeqDataset
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.graphmixer import GraphMixer
from jittor_geometric.datasets import JODIEDataset
from jittor_geometric.evaluate import MRR_Evaluator
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
from jittor_geometric.nn.dense.merge_predictor import MergeLayer
def test(loader):
    mrr_eval = MRR_Evaluator()
    model.eval()
    res_list = {}
    ap_list, auc_list, mrr_list = [], [], []
    loader_tqdm = tqdm(loader, ncols=120)
    for _, batch_data in enumerate(loader_tqdm):
        src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
        # compute the embeddings of src, dst, and neg_dst nodes
        src_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=src, node_interact_times=t,num_neighbors=num_neighbors)
        dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=dst, node_interact_times=t,num_neighbors=num_neighbors)
        neg_dst_node_embeddings = model[0].compute_node_temporal_embeddings(node_ids=neg_dst, node_interact_times=t,num_neighbors=num_neighbors)
        # compute the scores of positive and negative samples
        pos_score = jt.sigmoid(model[1](src_node_embeddings, dst_node_embeddings)).cpu().numpy()
        neg_score = jt.sigmoid(model[1](src_node_embeddings, neg_dst_node_embeddings)).cpu().numpy()
        num_neg = neg_score.shape[0]//pos_score.shape[0]
        if num_neg == 1: # ap
            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score])
            ap = average_precision_score(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
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
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_idx_data_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):
            src, dst, t, neg_dst = batch_data.src, batch_data.dst, batch_data.t, batch_data.neg_dst
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
            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
        train_loss = np.mean(train_losses)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        ap = test(val_loader)
        print(f'Epoch: {epoch:03d}, Val: {ap}')
        if ap['AP'] > best_ap:
            best_ap = ap['AP']
            jt.save(model.state_dict(), f'{save_model_path}/{dataset_name}_GraphMixer.pkl')
        else:
            patience -= 1
            if patience == 0:
                break

node_feat_dims = 172
edge_feat_dims = 172
hidden_dims = 100
num_layers = 2
jt.flags.use_cuda = 1 #jt.has_cuda
num_epochs = 1
time_feat_dim = 100
num_neighbors = 30
dropout = 0.1
save_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'saved_models')
if not osp.exists(save_model_path):
    os.makedirs(save_model_path)
criterion = jt.nn.BCEWithLogitsLoss()
dataset_name = 'wikipedia'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
if not osp.exists(path):
    os.makedirs(path)
if dataset_name in ['GoogleLocal', 'Yelp', 'Taobao', 'ML-20M' 'Flickr', 'YouTube', 'Patent', 'WikiLink']: # for TGBSeqDataset
    dataset = TGBSeqDataset(root=path, name=dataset_name)
    train_idx=np.nonzero(dataset.train_mask)[0]
    val_idx=np.nonzero(dataset.val_mask)[0]
    test_idx=np.nonzero(dataset.test_mask)[0]
    edge_ids=np.arange(dataset.num_edges)+1
    # test_ns is the negative samples for test set
    if dataset.test_ns is not None:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), test_ns=jt.array(dataset.test_ns.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    else:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    train_data, val_data, test_data = data.train_val_test_split_w_mask()
    train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1)
    val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
    test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)
elif dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm']: # for JODIEDataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
    dataset = JODIEDataset(path, name=dataset_name) # wikipedia, mooc, reddit, lastfm
    data = dataset[0]
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)

# Define the neighbor loader
full_neighbor_sampler = get_neighbor_sampler(data, 'recent',seed=0)

edge_raw_features = None
# The index of node and edge both start from 1
node_raw_features = np.zeros((data.num_nodes+1, node_feat_dims))
if isinstance(dataset, JODIEDataset):
    edge_raw_features = data.msg
else:
    if dataset.edge_feat is not None:
        edge_raw_features = jt.array(dataset.edge_feat)

if edge_raw_features is not None:
    if edge_raw_features.shape[0] != data.num_edges + 1:
        edge_raw_features = np.concatenate((np.zeros((1, edge_raw_features.shape[1])), edge_raw_features), axis=0)
    edge_feat_dims = edge_raw_features.shape[1]
else:
    edge_raw_features = np.zeros((data.num_edges+1, edge_feat_dims))
dynamic_backbone = GraphMixer(node_raw_features, edge_raw_features, full_neighbor_sampler,time_feat_dim=time_feat_dim, num_tokens=num_neighbors, num_layers=num_layers, dropout=dropout)
link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],hidden_dim=node_raw_features.shape[1], output_dim=1)
model = nn.Sequential(dynamic_backbone, link_predictor)

optimizer = jt.nn.Adam(list(model.parameters()),lr=0.0001)

train()
model.load_state_dict(jt.load(f'{save_model_path}/{dataset_name}_GraphMixer.pkl'))
print(test(test_loader))
