import time
import random
import argparse
import os
import os.path as osp
import traceback
import numpy as np
import scipy.sparse as sp
import jittor as jt
from jittor import nn
from jittor_geometric import transforms as T
from jittor_geometric.datasets import Planetoid
from jittor_geometric.nn import NAGphormerModel, accuracy_batch, re_features, laplacian_positional_encoding
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.utils import to_undirected


class NumpyDataset(jt.dataset.Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
        self.set_attrs(total_len=features.shape[0])

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
def get_dataset(dataset, pe_dim):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
    print('======load dataset {}'.format(args.dataset))

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    data = dataset[0]
    v_num = data.x.shape[0]
    edge_index, edge_weight = gcn_norm(
                        data.edge_index,  data.edge_attr,v_num,
                        improved=False, add_self_loops=True)

    adj = sp.coo_matrix(
        (edge_weight.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(v_num, v_num)
    )

    graph_edge_index, graph_edge_weight = to_undirected(data.edge_index, data.edge_attr)

    if graph_edge_weight is None:
        graph = sp.coo_matrix(
            (jt.ones(graph_edge_index.shape[1]), (graph_edge_index[0].numpy(), graph_edge_index[1].numpy())),
            shape=(v_num, v_num)
        )
    else:
        graph = sp.coo_matrix(
            (graph_edge_weight.numpy(), (graph_edge_index[0].numpy(), graph_edge_index[1].numpy())),
            shape=(v_num, v_num)
        )
    lpe = laplacian_positional_encoding(graph, pe_dim)

    features = jt.Var(data.x.numpy()).float32()
    # features = col_normalize(features)
    features = jt.cat((features, jt.Var(lpe).float32()), dim=1)
    labels = data.y.numpy()

    idx = np.arange(v_num)
    random.shuffle(idx)

    # 60% train, 20% val, 20% test
    train_size = int(0.6 * v_num)
    val_size = int(0.2 * v_num)
    
    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size + val_size]
    idx_test = idx[train_size + val_size:]
    idx_train = jt.Var(idx_train).int32()
    idx_val = jt.Var(idx_val).int32()
    idx_test = jt.Var(idx_test).int32()

    global nums
    nums = [len(idx_train), len(idx_val), len(idx_test)]
    
    if len(labels.shape) > 1: 
        labels = jt.argmax(jt.Var(labels), dim=-1)
    else:
        labels = jt.Var(labels)

    return adj, features, labels, idx_train, idx_val, idx_test



# Setup configuration
jt.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed')
parser.add_argument('--hops', type=int, default=3)
parser.add_argument('--pe_dim', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--ffn_dim', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--attention_dropout', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.00001)

args = parser.parse_args()

# Load dataset and create features with positional encoding
adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.pe_dim)

# Process features using hop-based aggregation
processed_features = re_features(adj, features.numpy(), args.hops) 
processed_features = jt.Var(processed_features).float32()

train_dataset = NumpyDataset(processed_features[idx_train], labels[idx_train])
val_dataset   = NumpyDataset(processed_features[idx_val],   labels[idx_val])
test_dataset  = NumpyDataset(processed_features[idx_test],  labels[idx_test])

# Create data loaders
train_loader = jt.dataset.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = jt.dataset.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True)
test_loader  = jt.dataset.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

# Initialize NAGphormer model and optimizer
model = NAGphormerModel(
    hops=args.hops,
    n_class=int(jt.max(labels).item() + 1),
    input_dim=features.shape[1],
    pe_dim=args.pe_dim,
    n_layers=args.n_layers,
    num_heads=args.n_heads,
    hidden_dim=args.hidden_dim,
    ffn_dim=args.ffn_dim,
    dropout_rate=args.dropout,
    attention_dropout_rate=args.attention_dropout,
)

optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training function
def train():
    model.train()
    loss_train_b = 0.0
    for batch in train_loader:
        nodes_features, batch_labels = batch
        output = model(nodes_features)
        if batch_labels.ndim > 1:
            batch_labels = batch_labels.squeeze()
        batch_labels = batch_labels.int32()

        loss_train = nn.nll_loss(output, batch_labels)
        optimizer.step(loss_train)
        loss_train_b += float(loss_train)

# Evaluation function
def test():
    global nums
    model.eval()
    acc_list, loss_list = [], []
    data_loader = [train_loader, val_loader, test_loader]
    for idx, loader in enumerate(data_loader):
        loss = 0
        acc = 0
        for batch in loader:
            nodes_features, batch_labels = batch
            output = model(nodes_features)
            if batch_labels.ndim > 1:
                batch_labels = batch_labels.squeeze()
            batch_labels = batch_labels.int32()
            loss += float(nn.nll_loss(output, batch_labels))
            acc += float(accuracy_batch(output, batch_labels))
        acc /= nums[idx]
        acc_list.append(acc)
        loss_list.append(loss)
    return acc_list[0], acc_list[1], acc_list[2]


best_val_acc = test_acc = 0
# Training loop
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
print("NAGphormer on dataset {}: Best Val Acc: {:.4f}, Test Acc: {:.4f}".format(
    args.dataset, best_val_acc, test_acc))