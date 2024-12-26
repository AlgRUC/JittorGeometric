import os.path as osp
import sys
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
import jittor as jt
from jittor.nn import Embedding, Linear, RNNCell
from jittor_geometric.datasets import JODIEDataset, TemporalDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from jittor_geometric.datasets.tgb_seq import TGBSeqDataset
from jittor_geometric.data import TemporalData

jt.flags.use_cuda = 1 # jt.has_cuda


dataset_name = 'GoogleLocal'# wikipedia, mooc, reddit, lastfm
if dataset_name in [ 'wikipedia', 'reddit', 'mooc', 'lastfm']:
    # Load the dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
    dataset = JODIEDataset(path, name=dataset_name) 
    data = dataset[0]

    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # Split the dataset into train/val/test
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    # Create TemporalDataLoader objects
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)

elif dataset_name in ['GoogleLocal', 'Yelp', 'Taobao', 'ML-20M' 'Flickr', 'YouTube', 'Patent', 'WikiLink']:
    path='/data/lu_yi/tgb-seq/'
    dataset = TGBSeqDataset(root=path, name=dataset_name)
    train_idx=np.nonzero(dataset.train_mask)[0]
    val_idx=np.nonzero(dataset.val_mask)[0]
    test_idx=np.nonzero(dataset.test_mask)[0]
    edge_ids=np.arange(dataset.num_edges)+1
    if dataset.test_ns is not None:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), msg=jt.array(dataset.edge_feat), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), test_ns=jt.array(dataset.test_ns.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    else:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), msg=jt.array(dataset.edge_feat), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    train_data, val_data, test_data = data.train_val_test_split_w_mask()
    train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1)
    val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
    test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)

def compute_src_dst_node_time_shifts(src_node_ids, dst_node_ids, node_interact_times):
    src_node_last_timestamps = {}
    dst_node_last_timestamps = {}
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []

    for i in range(len(src_node_ids)):
        src_node_id = src_node_ids[i]
        dst_node_id = dst_node_ids[i]
        node_interact_time = node_interact_times[i]

        if src_node_id not in src_node_last_timestamps:
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps:
            dst_node_last_timestamps[dst_node_id] = 0

        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])

        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time

    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift, dst_node_std_time_shift

src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift, dst_node_std_time_shift = compute_src_dst_node_time_shifts(
    src_node_ids=data.src.numpy(), 
    dst_node_ids=data.dst.numpy(), 
    node_interact_times=data.t.numpy()
)

class JODIEEmbedding(jt.nn.Module):
    def __init__(self, embedding_dim, num_users, num_items,
                 src_node_mean_time_shift, src_node_std_time_shift,
                 dst_node_mean_time_shift, dst_node_std_time_shift):
        super(JODIEEmbedding, self).__init__()
        
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)
        
        self.time_embedding = Linear(1, embedding_dim)
        
        self.user_rnn = RNNCell(embedding_dim + embedding_dim, embedding_dim)
        self.item_rnn = RNNCell(embedding_dim + embedding_dim, embedding_dim)
        
        self.last_user_timestamp = jt.ones(num_users) * -1
        self.last_item_timestamp = jt.ones(num_items) * -1
        
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift = dst_node_mean_time_shift
        self.dst_node_std_time_shift = dst_node_std_time_shift

    def execute(self, user_idx, item_idx, timestamp):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        last_user_time = self.last_user_timestamp[user_idx]
        last_item_time = self.last_item_timestamp[item_idx]
        delta_t_user = timestamp - last_user_time
        delta_t_item = timestamp - last_item_time

        delta_t_user = (delta_t_user - self.src_node_mean_time_shift) / (self.src_node_std_time_shift + 1e-9)
        delta_t_item = (delta_t_item - self.dst_node_mean_time_shift) / (self.dst_node_std_time_shift + 1e-9)

        delta_t_user = delta_t_user.unsqueeze(-1)
        delta_t_item = delta_t_item.unsqueeze(-1)
        time_emb_user = self.time_embedding(delta_t_user)
        time_emb_item = self.time_embedding(delta_t_item)

        user_input = jt.concat([user_emb, time_emb_user], dim=-1)
        item_input = jt.concat([item_emb, time_emb_item], dim=-1)

        user_emb = self.user_rnn(user_input, user_emb)
        item_emb = self.item_rnn(item_input, item_emb)

        self.last_user_timestamp[user_idx] = timestamp
        self.last_item_timestamp[item_idx] = timestamp

        return user_emb, item_emb


class LinkPredictor(jt.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def execute(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = jt.nn.relu(h)
        return self.lin_final(h)


embedding_dim = 20
num_users = int(data.src.max()) + 1
num_items = int(data.dst.max()) + 1

jodie_model = JODIEEmbedding(embedding_dim, num_users, num_items, 
                                    src_node_mean_time_shift, src_node_std_time_shift,
                                    dst_node_mean_time_shift, dst_node_std_time_shift)

predictor = LinkPredictor(embedding_dim)
optimizer = jt.nn.Adam(jodie_model.parameters() + predictor.parameters(), lr=0.0001)
criterion = jt.nn.BCEWithLogitsLoss()

def train():
    jodie_model.train()
    predictor.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        user_idx = batch.src
        item_idx = batch.dst
        timestamp = batch.t
        neg_item_idx = batch.neg_dst
        
        pos_user_emb, pos_item_emb = jodie_model(user_idx, item_idx, timestamp)
        neg_user_emb, neg_item_emb = jodie_model(user_idx, neg_item_idx, timestamp)

        pos_pred = predictor(pos_user_emb, pos_item_emb)
        neg_pred = predictor(neg_user_emb, neg_item_emb)

        loss = criterion(pos_pred, jt.ones_like(pos_pred)) + criterion(neg_pred, jt.zeros_like(neg_pred))

        optimizer.step(loss)
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


def test(loader):
    jodie_model.eval()
    predictor.eval()
    aps, aucs = [], []
    for batch in loader:
        user_idx = batch.src
        item_idx = batch.dst
        timestamp = batch.t
        neg_item_idx = jt.randint(0, num_items, (user_idx.shape[0],))

        pos_user_emb, pos_item_emb = jodie_model(user_idx, item_idx, timestamp)
        neg_user_emb, neg_item_emb = jodie_model(user_idx, neg_item_idx, timestamp)

        pos_pred = predictor(pos_user_emb, pos_item_emb)
        neg_pred = predictor(neg_user_emb, neg_item_emb)

        y_pred = jt.concat([pos_pred.sigmoid(), neg_pred.sigmoid()], dim=0).numpy()
        y_true = jt.concat([jt.ones(pos_pred.shape[0]), jt.zeros(neg_pred.shape[0])], dim=0).numpy()

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

    return np.mean(aps), np.mean(aucs)


for epoch in range(1, 11):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')