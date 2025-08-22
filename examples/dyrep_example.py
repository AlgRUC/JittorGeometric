import os.path as osp
import sys
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
import jittor as jt
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor.nn import Linear
from jittor_geometric.datasets import JODIEDataset, TemporalDataLoader
from jittor_geometric.nn import DyRepMemory, TransformerConv
from jittor_geometric.nn.models.dyrep import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tqdm import *
import numpy as np
from jittor_geometric.datasets.tgb_seq import TGBSeqDataset
from jittor_geometric.data import TemporalData

# Setup configuration
jt.flags.use_cuda = 1

# Parse arguments and load dataset
import argparse
parser = argparse.ArgumentParser(description='Train DyRep model on specified dataset.')
parser.add_argument('--dataset_name', type=str, default='wikipedia',
                    help='Name of the dataset (wikipedia, mooc, reddit, lastfm). Default: wikipedia')
args = parser.parse_args()
dataset_name = args.dataset_name
print('dataset_name:', args.dataset_name)

# Load dataset based on type
if dataset_name in [ 'wikipedia', 'reddit', 'mooc', 'lastfm']:
    # Load dataset from JODIE
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
    dataset = JODIEDataset(path, name=dataset_name) 
    data = dataset[0]
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    
    # Split the dataset into train/val/test sets
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    
    # Create data loaders
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)
elif dataset_name in ['GoogleLocal', 'Yelp', 'Taobao', 'ML-20M' 'Flickr', 'YouTube', 'WikiLink']:
    # Load dataset from TGB-Seq
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = TGBSeqDataset(root=path, name=dataset_name)
    train_idx=np.nonzero(dataset.train_mask)[0]
    val_idx=np.nonzero(dataset.val_mask)[0]
    test_idx=np.nonzero(dataset.test_mask)[0]
    edge_ids=np.arange(dataset.num_edges)+1
    if dataset.edge_feat is None:
        edge_feat=np.zeros((dataset.num_edges+1, 172))
    else:
        edge_feat=dataset.edge_feat
    if dataset.test_ns is not None:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), msg=jt.array(edge_feat), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), test_ns=jt.array(dataset.test_ns.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    else:
        data = TemporalData(src=jt.array(dataset.src_node_ids.astype(np.int32)), dst=jt.array(dataset.dst_node_ids.astype(np.int32)), t=jt.array(dataset.time), msg=jt.array(edge_feat), train_mask=jt.array(train_idx.astype(np.int32)), val_mask=jt.array(val_idx.astype(np.int32)), test_mask=jt.array(test_idx.astype(np.int32)), edge_ids=jt.array(edge_ids.astype(np.int32)))
    # Split the dataset into train/val/test sets
    train_data, val_data, test_data = data.train_val_test_split_w_mask()
    
    # Create TemporalDataLoader objects
    train_loader = TemporalDataLoader(train_data, batch_size=200, num_neg_sample=1)
    val_loader = TemporalDataLoader(val_data, batch_size=200, num_neg_sample=1)
    test_loader = TemporalDataLoader(test_data, batch_size=200, num_neg_sample=1)

# Initialize neighbor loader
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10)

# Graph attention embedding module
class GraphAttentionEmbedding(jt.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def execute(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t)
        edge_attr = jt.concat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr) 

# MLP-based link predictor
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

# Define Memory module
memory_dim = time_dim = embedding_dim = 100
memory = DyRepMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
)

link_pred = LinkPredictor(in_channels=embedding_dim)

model = jt.nn.Sequential(memory, gnn, link_pred)
optimizer = jt.nn.Adam(list(model.parameters()),lr=0.0001)
criterion = jt.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones
assoc = jt.empty(data.num_nodes, dtype=jt.int32)


def train():
    model.train()
    model[0].reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    for batch in tqdm(train_loader):
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = jt.arange(n_id.size(0))
        
        # Get updated memory of all nodes involved in the computation
        z, last_update = model[0](n_id)
        z = model[1](z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        # Compute predictions and loss
        pos_out = model[2](z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = model[2](z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        loss = criterion(pos_out, jt.ones_like(pos_out))
        loss += criterion(neg_out, jt.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state
        model[0].update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        # Backpropagation and optimization
        optimizer.zero_grad()
        optimizer.step(loss)
        model[0].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


def test(loader):
    model.eval()

    # Ensure deterministic sampling across epochs
    jt.set_seed(12345)

    aps, aucs = [], []
    for batch in loader:
        src, pos_dst, t, msg = batch['src'], batch['dst'], batch['t'], batch['msg']
        neg_dst = jt.randint(min_dst_idx, max_dst_idx + 1, (src.shape[0],), dtype=jt.int32)

        n_id = jt.concat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = jt.arange(n_id.shape[0])

        # Get updated memory of all nodes involved in the computation
        z, last_update = model[0](n_id)
        z = model[1](z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        # Compute predictions
        pos_out = model[2](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model[2](z[assoc[src]], z[assoc[neg_dst]])
        y_pred = jt.concat([pos_out, neg_out], dim=0).sigmoid().numpy()
        y_true = jt.concat([jt.ones(pos_out.shape[0]), jt.zeros(neg_out.shape[0])], dim=0).numpy()

        # Compute metrics
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        model[0].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(jt.Var(aps).mean()), float(jt.Var(aucs).mean())

best_ap = 0
patience = 5
save_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'saved_models')
for epoch in range(1, 6):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    if val_ap > best_ap:
        best_ap = val_ap
        # Save the model when achieving better performance on val set
        jt.save(model.state_dict(), f'{save_model_path}/{dataset_name}_model_DyRep.pkl')
        print('Saved model is updated')
        patience = 5
    else:
        patience -= 1
        # Early stop if patience decreases to zero
        if patience == 0:
            break

# Load the saved model for testing
model.load_state_dict(jt.load(f'{save_model_path}/{dataset_name}_model_DyRep.pkl'))
test_ap, test_auc = test(test_loader)
print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')