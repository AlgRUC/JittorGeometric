import os.path as osp
import sys
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
import jittor as jt
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor.nn import Linear
from jittor_geometric.datasets import JODIEDataset, TemporalDataLoader
from jittor_geometric.nn import TGNMemory, TransformerConv
from jittor_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tqdm import *

jt.flags.use_cuda = 1 #jt.has_cuda

# Load the dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia') # wikipedia, mooc, reddit, lastfm
data = dataset[0]

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# Split the dataset into train/val/test
train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

# Create TemporalDataLoader objects
train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0)
test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0)

# Define the neighbor loader
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10)

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


memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
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

optimizer = jt.nn.Adam(
    list(memory.parameters()) + list(gnn.parameters()) + list(link_pred.parameters()), lr=0.0001)

criterion = jt.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = jt.empty(data.num_nodes, dtype=jt.int32)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = jt.arange(n_id.size(0))
        
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        # Compute predictions and loss.
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        
        loss = criterion(pos_out, jt.ones_like(pos_out))
        loss += criterion(neg_out, jt.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        

        # Backpropagation and optimization.
        optimizer.step(loss)
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    jt.set_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        src, pos_dst, t, msg = batch['src'], batch['dst'], batch['t'], batch['msg']
        neg_dst = jt.randint(min_dst_idx, max_dst_idx + 1, (src.shape[0],), dtype=jt.int32)

        n_id = jt.concat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = jt.arange(n_id.shape[0])

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = jt.concat([pos_out, neg_out], dim=0).sigmoid().numpy()
        y_true = jt.concat([jt.ones(pos_out.shape[0]), jt.zeros(neg_out.shape[0])], dim=0).numpy()

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(jt.Var(aps).mean()), float(jt.Var(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')