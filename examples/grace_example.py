import argparse
import warnings
import time
import jittor as jt
import jittor.nn as nn
import numpy as np
from jittor_geometric.datasets import Planetoid, WikipediaNetwork, GeomGCN
import jittor_geometric.transforms as T
from jittor_geometric.nn.models.grace import Grace
from jittor_geometric.utils.gssl_utils import random_splits, set_seed, aug
import seaborn as sns
from alive_progress import alive_bar
import random

from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

warnings.filterwarnings("ignore")

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

def count_parameters(model):
    return sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--epochs", type=int, default=500, help="Number of training periods."
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd2', type=float, default=0, help='Weight decay of linear evaluator.')
parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")
parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument(
    "--hid_dim", type=int, default=512, help="Hidden layer dim."
)
parser.add_argument(
    "--out_dim", type=int, default=512, help="Output layer dim."
)
parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of GNN layers."
)
parser.add_argument(
    "--der1",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 1st augmentation.",
)
parser.add_argument(
    "--der2",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 2nd augmentation.",
)
parser.add_argument(
    "--dfr1",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 1st augmentation.",
)
parser.add_argument(
    "--dfr2",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 2nd augmentation.",
)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()

if args.gpu != -1 and jt.has_cuda:
    jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 0

set_seed(args.seed)

if __name__ == "__main__":
    # Step 1: Load hyperparameters =================================================================== #
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = {"relu": nn.ReLU(), "prelu": nn.PReLU()}[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    dataset = DataLoader(name=args.dataname)
    data = dataset[0]
    feat = data.x
    label = data.y

    n_node = feat.shape[0]
    num_class = dataset.num_classes

    edge_index, edge_weight = data.edge_index, data.edge_attr
    in_dim = feat.shape[1]

    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, n_node,
        improved=False, add_self_loops=True)
    with jt.no_grad():
        data.csc = cootocsc(edge_index, edge_weight, n_node)
        data.csr = cootocsr(edge_index, edge_weight, n_node)

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    print(f"# params: {count_parameters(model)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params / 1024} K")
    optimizer = jt.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = float("inf")
    cnt_wait = 0
    time_run = []

    # Step 4: Training =======================================================================
    with alive_bar(epochs) as bar:
        for epoch in range(epochs):
            t_st = time.time()

            model.train()
            optimizer.zero_grad()
            graph1, feat1 = aug(data, feat, drop_feature_rate_1, drop_edge_rate_1)
            graph2, feat2 = aug(data, feat, drop_feature_rate_2, drop_edge_rate_2)

            loss = model(graph1, graph2, feat1, feat2)
            optimizer.backward(loss)
            optimizer.step()
            time_epoch = time.time() - t_st  # each epoch train times
            time_run.append(time_epoch)
            if epoch % 20 == 0:
                print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
            if loss < best:
                best = loss
                cnt_wait = 0
                jt.save(model.state_dict(), "model_grace.pkl")
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")
    run_sum = sum(time_run)
    epochsss = len(time_run)

    print("each run avg_time:", run_sum, "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    # Evaluation
    label = label
    feat = feat
    model.load_state_dict(jt.load("model_grace.pkl"))
    model.eval()
    embeds = model.get_embedding(data, feat)

    """Evaluation Embeddings"""
    print("=== Load ===")
    print(embeds)
    print(embeds.shape)
    print(jt.norm(embeds))
    print("=== Evaluation ===")
    ''' Linear Evaluation '''
    results = []
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042, 2424918363]
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(label) / num_class))
    val_lb = int(round(val_rate * len(label)))
    for i in range(10):
        seed = SEEDS[i]
        train_mask, val_mask, test_mask = random_splits(label, num_class, percls_trn, val_lb, seed=seed)

        train_embs = embeds[train_mask]
        val_embs = embeds[val_mask]
        test_embs = embeds[test_mask]

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=out_dim, n_classes=num_class)
        opt = jt.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = jt.argmax(logits, dim=1)[0]
            train_acc = jt.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            opt.backward(loss)
            opt.step()

            logreg.eval()
            with jt.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = jt.argmax(val_logits, dim=1)[0]
                test_preds = jt.argmax(test_logits, dim=1)[0]

                val_acc = jt.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = jt.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        print(i, 'Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        results.append(eval_acc.numpy())

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')
    # python grace_example.py --dataname cora --lr 5e-4 --wd 1e-5 --act_fn relu --der1 0.2 --der2 0.4 --dfr1 0.3 --dfr2 0.4 --temp 0.4 --gpu 1 --patience 50