import argparse
import time
from alive_progress import alive_bar
import random
import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor_geometric.data import Data
from jittor_geometric.utils import add_self_loops
from jittor_geometric.datasets import Planetoid, WikipediaNetwork, GeomGCN
import jittor_geometric.transforms as T

from jittor_geometric.nn.models.dgi import DGI
from jittor_geometric.utils.gssl_utils import random_splits, set_seed
import seaborn as sns
# Data preprocessing utilities
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

# Logistic regression classifier for evaluation
class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, n_classes)

    def execute(self, x):
        ret = self.fc(x)
        return ret

# Dataset loader for various graph datasets
def DataLoader(name):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='../../data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='../../data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell', 'wisconsin', 'actor']:
        dataset = GeomGCN(root='../../data/', name=name)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset

# Parse arguments
parser = argparse.ArgumentParser(description="DGI")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--dropout", type=float, default=0.0, help="dropout probability")
parser.add_argument("--dataname", type=str, default="cora", help="Name of dataset.")
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--dgi-lr", type=float, default=1e-3, help="dgi learning rate")
parser.add_argument("--classifier-lr", type=float, default=1e-2, help="classifier learning rate")
parser.add_argument("--n-dgi-epochs", type=int, default=300, help="number of training epochs")
parser.add_argument("--wd2", type=float, default=0, help="Weight decay of linear evaluator.")
parser.add_argument("--n-hidden", type=int, default=512, help="number of hidden gcn units")
parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight for L2 loss")
parser.add_argument("--patience", type=int, default=20, help="early stop patience condition")
parser.add_argument("--self-loop", action="store_true", help="graph self-loop (default=False)")
parser.add_argument("--dev", type=int, default=0, help="device id")

parser.set_defaults(self_loop=False)
args = parser.parse_args()

# Setup configuration
if args.gpu != -1 and jt.has_cuda:
    jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 0

set_seed(args.seed)

def main(args):
    # Load dataset and prepare data
    dataset = DataLoader(name=args.dataname)
    data = dataset[0]
    features = data.x
    label = data.y
    n_node = features.shape[0]
    num_features = dataset.num_features
    n_classes = dataset.num_classes
    in_feats = features.shape[1]

    edge_index, edge_weight = data.edge_index, data.edge_attr

    # Prepare edge normalization
    if args.self_loop:
        edge_index, _ = add_self_loops(edge_index)
    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, n_node,
        improved=False, add_self_loops=True)
    # Convert to sparse matrix format
    with jt.no_grad():
        data.csc = cootocsc(edge_index, edge_weight, n_node)
        data.csr = cootocsr(edge_index, edge_weight, n_node)

    # Initialize DGI model and optimizer
    dgi = DGI(
        data,
        in_feats,
        args.n_hidden,
        args.n_layers,
        args.dropout,
    )

    total_params = sum(p.numel() for p in dgi.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params / 1024} K")

    dgi_optimizer = jt.optim.Adam(
        dgi.parameters(), lr=args.dgi_lr, weight_decay=args.weight_decay
    )

    # Training loop for DGI
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    time_run = []

    with alive_bar(args.n_dgi_epochs) as bar:
        for epoch in range(args.n_dgi_epochs):
            t_st = time.time()

            dgi.train()
            dgi_optimizer.zero_grad()
            loss = dgi(features)

            dgi_optimizer.step(loss)

            time_epoch = time.time() - t_st
            time_run.append(time_epoch)

            if loss.item() < best:
                best = loss.item()
                best_t = epoch
                cnt_wait = 0
                jt.save(dgi.state_dict(), "best_dgi.pkl")
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping!")
                break

            if epoch >= 3:
                dur.append(time.time() - t_st)

            if epoch % 20 == 0 and epoch > 0:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                    "ETputs(KTEPS) {:.2f}".format(
                        epoch, np.mean(dur), loss.item(), edge_index.shape[1] / np.mean(dur) / 1000
                    )
                )
            bar()

    run_sum = sum(time_run)
    epochsss = len(time_run)

    print("Each run avg_time:", run_sum, "s")
    print("Each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    # Load best model and generate embeddings
    print("Loading {}th epoch".format(best_t))
    dgi.load_state_dict(jt.load("best_dgi.pkl"))
    dgi.eval()
    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()
    
    # Linear evaluation with multiple splits
    print("=== Evaluation ===")
    results = []
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(label) / n_classes))
    val_lb = int(round(val_rate * len(label)))

    for i in range(10):
        seed = SEEDS[i]
        assert len(label) == n_node
        train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

        train_mask = jt.bool(train_mask)
        val_mask = jt.bool(val_mask)
        test_mask = jt.bool(test_mask)

        train_embs = embeds[train_mask]
        val_embs = embeds[val_mask]
        test_embs = embeds[test_mask]

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=args.n_hidden, n_classes=n_classes)
        opt = nn.Adam(logreg.parameters(), lr=args.classifier_lr, weight_decay=args.wd2)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()

            logits = logreg(train_embs)
            preds = jt.argmax(logits, dim=1)[0]
            train_acc = jt.sum(preds == train_labels).float32() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            opt.step(loss)

            logreg.eval()
            with jt.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = jt.argmax(val_logits, dim=1)[0]
                test_preds = jt.argmax(test_logits, dim=1)[0]

                val_acc = jt.sum(val_preds == val_labels).float32() / val_labels.shape[0]
                test_acc = jt.sum(test_preds == test_labels).float32() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        print(i, 'Linear evaluation accuracy:{:.4f}'.format(eval_acc.item()))
        results.append(eval_acc)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000)) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')


if __name__ == "__main__":
    print(args)
    main(args)
    # python dgi_example.py --dataname cora