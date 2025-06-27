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
from jittor_geometric.nn.models.sasrec import SASRec
from jittor_geometric.datasets import JODIEDataset
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
from jittor_geometric.utils.bprloss import BPRLoss

def test(loader):
    mrr_eval = MRR_Evaluator()
    model.eval()
    res_list = {}
    ap_list, auc_list, mrr_list = [], [], []
    loader_tqdm = tqdm(loader, ncols=120)
    for _, batch_data in enumerate(loader_tqdm):
        src, dst, t, neg_dst = jt.array(batch_data.src), jt.array(batch_data.dst), jt.array(batch_data.t), jt.array(batch_data.neg_dst)
        src_neighb_seq, _, _ = full_neighbor_sampler.get_historical_neighbors_left(node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
        neighbor_num=(src_neighb_seq!=0).sum(axis=1)
        if neighbor_num.sum() == 0:
            continue
        pos_item = jt.Var(dst).unsqueeze(-1)
        neg_item = jt.Var(neg_dst).unsqueeze(-1)
        test_dst = jt.cat([pos_item, neg_item], dim=-1)
        batch_data=[jt.Var(src_neighb_seq), jt.Var(neighbor_num), jt.Var(test_dst)]
        pos_score, neg_score = model.predict(batch_data)
        neg_score = neg_score.flatten()
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
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_idx_data_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):
            src, dst, t, neg_dst = jt.array(batch_data.src), jt.array(batch_data.dst), jt.array(batch_data.t), jt.array(batch_data.neg_dst)
            src_neighb_seq, _, _ = full_neighbor_sampler.get_historical_neighbors_left(node_ids=src.numpy(), node_interact_times=t.numpy(), num_neighbors=num_neighbors)
            neighbor_num=(src_neighb_seq!=0).sum(axis=1)
            if neighbor_num.sum() == 0:
                continue
            pos_item = jt.Var(dst)
            neg_item = jt.Var(neg_dst)
            test_dst = jt.cat([pos_item, neg_item], dim=0)
            batch_data=[jt.Var(src_neighb_seq), jt.Var(neighbor_num), jt.Var(test_dst)]
            batch_src_node_embeddings, dst_node_embeddings = model.calculate_loss(batch_data)
            batch_dst_node_embeddings = dst_node_embeddings[:len(pos_item)]
            batch_neg_dst_node_embeddings = dst_node_embeddings[len(pos_item):]
            batch_neg_src_node_embeddings = batch_src_node_embeddings
            logits_pos = jt.matmul(batch_dst_node_embeddings, batch_neg_src_node_embeddings.transpose(0, 1))
            logits_neg = jt.matmul(batch_neg_dst_node_embeddings, batch_neg_src_node_embeddings.transpose(0, 1))
            loss = loss_func(logits_pos, logits_neg)
            optimizer.zero_grad()
            optimizer.step(loss)
            train_losses.append(loss.item())
            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
        train_loss = np.mean(train_losses)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        ap = test(val_loader)
        print(f'Epoch: {epoch:03d}, Val: {ap}')
        # save the best model if AP is improved
        if ap['AP'] > best_ap:
            best_ap = ap['AP']
            jt.save(model.state_dict(), f'{save_model_path}/{dataset_name}_SASRec.pkl')
        else:
            patience -= 1
            if patience == 0:
                break

hidden_dims = 64
num_layers = 2
jt.flags.use_cuda = 1 #jt.has_cuda
num_epochs = 100
num_neighbors = 30
dropout = 0.1
save_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'saved_models')
if not osp.exists(save_model_path):
    os.makedirs(save_model_path)
criterion = BPRLoss()
dataset_name = 'wikipedia'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
if not osp.exists(path):
    os.makedirs(path)
if dataset_name in ['GoogleLocal', 'Yelp', 'Taobao', 'ML-20M' 'Flickr', 'YouTube', 'WikiLink']: # for TGBSeqDataset
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
    train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1, shuffle=True)
    val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
    test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)
elif dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm']: # for JODIEDataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
    dataset = JODIEDataset(path, name=dataset_name)
    data = dataset[0]
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0, shuffle=True)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)

# Define the neighbor loader
full_neighbor_sampler = get_neighbor_sampler(data, 'recent',seed=1)
if dataset_name in ['GoogleLocal', 'ML-20M', 'Taobao', 'Yelp', 'mooc', 'lastfm', 'reddit', 'wikipedia']:
    user_size = data.src_size
    item_size = data.dst_size
    node_size = data.max_node_id
else:
    user_size = data.max_node_id
    item_size = data.max_node_id
    node_size = data.max_node_id
dst_min_idx = data.dst.min()
src_min_idx = data.src.min()
hidden_size = 64
model = SASRec(n_layers=num_layers, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.1, attn_dropout_prob=0.1, hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, n_items=item_size, max_seq_length=num_neighbors)
model.set_min_idx(src_min_idx, dst_min_idx)
loss_func = BPRLoss()
layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
optimizer = jt.nn.Adam(list(model.parameters()),lr=0.0001)

train()
model.load_state_dict(jt.load(f'{save_model_path}/{dataset_name}_SASRec.pkl'))
print(test(test_loader))
