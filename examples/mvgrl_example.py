import argparse
import warnings
import seaborn as sns
import time
import random
import numpy as np
import jittor as jt
from jittor import nn
from alive_progress import alive_bar

warnings.filterwarnings("ignore")

from jittor_geometric.datasets import Planetoid, WikipediaNetwork, GeomGCN
import jittor_geometric.transforms as T
from jittor_geometric.nn.models.mvgrl import MVGRL
from jittor_geometric.utils.gssl_utils import set_seed, \
    random_splits, preprocess_features, compute_ppr
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.preprocessing import MinMaxScaler
from jittor_geometric.utils import add_self_loops

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, n_classes)

    def execute(self, x):
        ret = self.fc(x)
        return ret

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

def process_dataset(name, epsilon, preprocess=False, self_loop=False):
    dataset = DataLoader(name=name)
    data = dataset[0]

    edge_index = data.edge_index
    edge_weight = data.edge_attr
    # print('edge index:', edge_index)
    # print('edge weight:', edge_weight)

    feat = data.x.numpy()
    label = data.y.numpy()
    num_nodes = data.num_nodes

    print("computing ppr")
    diff_adj = compute_ppr(edge_index, num_nodes=num_nodes, alpha=0.2, self_loop=True)
    print("computing end")

    if preprocess:
        print("additional processing")
        feat = preprocess_features(feat)
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

    diff_edges = np.nonzero(diff_adj)
    diff_edge_index = np.vstack(diff_edges)  # shape: [2, num_edges]
    diff_weight = diff_adj[diff_edges]

    if self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        diff_edge_index, _ = add_self_loops(diff_edge_index, num_nodes=num_nodes)

    return (
        edge_index,
        edge_weight,
        diff_edge_index,
        feat,
        label,
        diff_weight,
    )

# para parser
parser = argparse.ArgumentParser(description="mvgrl")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument("--dataname", type=str, default="cora", help="Name of dataset.")
parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument("--patience", type=int, default=20, help="Patient epochs to wait before early stopping.")
parser.add_argument("--lr1", type=float, default=0.001, help="Learning rate of mvgrl.")
parser.add_argument("--lr2", type=float, default=0.01, help="Learning rate of linear evaluator.")
parser.add_argument("--wd1", type=float, default=0.0, help="Weight decay of mvgrl.")
parser.add_argument("--wd2", type=float, default=0.0, help="Weight decay of linear evaluator.")
parser.add_argument("--epsilon", type=float, default=0.01, help="Edge mask threshold of diffusion graph.")
parser.add_argument("--hid_dim", type=int, default=512, help="Hidden layer dim.")
parser.add_argument("--self-loop", action="store_true", help="graph self-loop (default=False)")
parser.add_argument("--preprocess", action="store_true", help="graph preprocess (default=False)")

args = parser.parse_args()

if args.gpu != -1 and jt.has_cuda:
    jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 0

set_seed(args.seed)

if __name__ == "__main__":
    print(args)
    # Step 1: Prepare data =================================================================== #
    (
        edge_index,
        edge_weight,
        diff_edge_index,
        feat,
        label,
        diff_weight,
    ) = process_dataset(args.dataname, args.epsilon, preprocess=args.preprocess, self_loop=args.self_loop)

    n_feat = feat.shape[1]
    n_classes = len(np.unique(label))

    feat = jt.array(feat).float32()
    diff_weight = jt.array(diff_weight).float32()
    label = jt.array(label).int32()

    n_node = feat.shape[0]
    lbl1 = jt.ones(n_node * 2)
    lbl2 = jt.zeros(n_node * 2)
    lbl = jt.concat((lbl1, lbl2))
    # print('edge_index, edge_weight',edge_index, edge_weight)
    # print(edge_index.dtype)
    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, n_node,
        improved=False, add_self_loops=True)
    with jt.no_grad():
        csc_o = cootocsc(edge_index, edge_weight, n_node)
        csr_o = cootocsr(edge_index, edge_weight, n_node)
    diff_edge_index = jt.array(diff_edge_index, dtype=jt.int32)
    # print('diff_edge_index, diff_weight',diff_edge_index, diff_weight)
    # print(diff_edge_index.dtype)
    diff_edge_index, diff_weight = gcn_norm(
        diff_edge_index, diff_weight, n_node,
        improved=False, add_self_loops=True)
    with jt.no_grad():
        csc_d = cootocsc(diff_edge_index, diff_weight, n_node)
        csr_d = cootocsr(diff_edge_index, diff_weight, n_node)

    # Step 2: Create model =================================================================== #
    model = MVGRL(n_feat, args.hid_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params / 1024} K")

    # Step 3: Create training components ===================================================== #
    optimizer = nn.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    time_run = []
    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            t_st = time.time()
            model.train()
            optimizer.zero_grad()

            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat[shuf_idx]

            out = model(csc_o, csr_o, feat, csc_d, csr_d, shuf_feat)
            loss = loss_fn(out, lbl)

            optimizer.step(loss)
            time_epoch = time.time() - t_st
            time_run.append(time_epoch)

            if epoch % 20 == 0:
                print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

            if loss < best:
                best = loss
                cnt_wait = 0
                jt.save(model.state_dict(), "model_mvgrl.pkl")  # save model
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()

    run_sum = sum(time_run)
    epochsss = len(time_run)
    print("each run avg_time:", run_sum, "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    # load model
    model.load_state_dict(jt.load("model_mvgrl.pkl"))
    model.eval()
    embeds = model.get_embedding(csc_o, csr_o, feat, csc_d, csr_d)

    # Step 5: Linear evaluation ========================================================== #
    print("=== Load ===")
    print(embeds)
    print(embeds.shape)
    print(jt.norm(embeds))
    print("=== Evaluation ===")
    ''' Linear Evaluation '''
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

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = nn.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

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
    # python mvgrl_example.py --dataname cora --preprocess
