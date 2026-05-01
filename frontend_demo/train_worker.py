#!/usr/bin/env python3
import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    status_file = os.path.join(frontend_dir, 'training_status.json')

    def init_status():
        status = {
            'running': True,
            'epoch': 0,
            'total_epochs': args.epochs,
            'loss': 0,
            'acc': 0,
            'error': None,
            'history': {'loss': [], 'acc': []},
            'finished': False
        }
        with open(status_file, 'w') as f:
            json.dump(status, f)
        return status

    def update_status(status):
        new_status = {
            'running': status['running'],
            'epoch': status['epoch'],
            'total_epochs': args.epochs,
            'loss': status['loss'],
            'acc': status['acc'],
            'error': status['error'],
            'history': status['history'],
            'finished': status['finished']
        }
        with open(status_file, 'w') as f:
            json.dump(new_status, f)
        print(f"[DEBUG] Status saved: epoch={new_status['epoch']}, loss={new_status['loss']:.4f}, acc={new_status['acc']:.4f}, history_loss_len={len(new_status['history']['loss'])}, history_acc_len={len(new_status['history']['acc'])}")

    status = init_status()

    try:
        print(f"[DEBUG] Starting real training for {args.model} on {args.dataset}")

        supported_models = ['GCN', 'GAT', 'GraphSAGE', 'ChebNet2', 'SGC', 'APPNP', 'GPRGNN', 'BernNet']
        if args.model not in supported_models:
            raise ValueError(f"Model {args.model} not supported. Supported models: {supported_models}")

        import jittor as jt
        from jittor import nn
        from math import log
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jittor_geometric.datasets import Planetoid
        from jittor_geometric.nn import GCNConv, GATConv, SAGEConv, ChebNetII, SGConv, APPNP, GPRGNN, BernNet
        from jittor_geometric.ops import cootocsr, cootocsc
        from jittor_geometric.nn.conv.gcn_conv import gcn_norm
        from jittor_geometric.nn.conv.sage_conv import sage_norm
        from jittor_geometric.utils import get_laplacian, add_self_loops
        from jittor_geometric.utils import add_remaining_self_loops
        from jittor_geometric.utils.num_nodes import maybe_num_nodes
        import jittor_geometric.transforms as T
        from jittor import Var

        jt.flags.use_cuda = 0
        jt.flags.lazy_execution = 0
        print(f"[DEBUG] CUDA disabled for stability")

        print(f"[DEBUG] Loading dataset: {args.dataset}")
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        dataset_name_lower = args.dataset.lower()
        
        if dataset_name_lower in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=data_root, name=args.dataset, transform=T.NormalizeFeatures())
        else:
            raise ValueError(f"Dataset {args.dataset} not supported. Supported datasets: Cora, Citeseer, Pubmed")

        data = dataset[0]
        print(f"[DEBUG] Dataset loaded: num_nodes={data.num_nodes}, num_features={dataset.num_features}, num_classes={dataset.num_classes}")

        v_num = data.x.shape[0]
        e_num = data.edge_index.shape[1]
        edge_index, edge_weight = data.edge_index, data.edge_attr

        def gcn_norm_func(edge_index, edge_weight=None, num_nodes=None, improved=False,
                        add_self_loops=True, dtype=None):
            fill_value = 2. if improved else 1.
            if isinstance(edge_index, Var):
                num_nodes = maybe_num_nodes(edge_index, num_nodes)
                if edge_weight is None:
                    edge_weight = jt.ones((edge_index.size(1), ))
                if add_self_loops:
                    edge_index, tmp_edge_weight = add_remaining_self_loops(
                        edge_index, edge_weight, fill_value, num_nodes)
                    assert tmp_edge_weight is not None
                    edge_weight = tmp_edge_weight
                row, col = edge_index[0], edge_index[1]
                shape = list(edge_weight.shape)
                shape[0] = num_nodes
                deg = jt.zeros(shape)
                deg = jt.scatter(deg, 0, col, src=edge_weight, reduce='add')
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
                return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            return edge_index, edge_weight

        if args.model == 'ChebNet2':
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=data.x.dtype, num_nodes=v_num)
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1.0, num_nodes=v_num)
            with jt.no_grad():
                data.csc = cootocsc(edge_index, edge_weight, v_num)
                data.csr = cootocsr(edge_index, edge_weight, v_num)
        elif args.model == 'GraphSAGE':
            edge_index, edge_weight = sage_norm(
                edge_index, edge_weight, v_num,
                improved=False, add_self_loops=True)
            with jt.no_grad():
                data.csc = cootocsc(edge_index, edge_weight, v_num)
                data.csr = cootocsr(edge_index, edge_weight, v_num)
        elif args.model == 'BernNet':
            # Compute Laplacian matrices for BernNet
            edge_index1, edge_weight1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=data.x.dtype, num_nodes=v_num)
            edge_index2, edge_weight2 = add_self_loops(edge_index1, -edge_weight1, fill_value=2., num_nodes=v_num)
            with jt.no_grad():
                data.csc1 = cootocsc(edge_index1, edge_weight1, v_num)
                data.csr1 = cootocsr(edge_index1, edge_weight1, v_num)
                data.csc2 = cootocsc(edge_index2, edge_weight2, v_num)
                data.csr2 = cootocsr(edge_index2, edge_weight2, v_num)
        else:
            edge_index, edge_weight = gcn_norm_func(
                edge_index, edge_weight, v_num,
                improved=False, add_self_loops=True)
            with jt.no_grad():
                data.csc = cootocsc(edge_index, edge_weight, v_num)
                data.csr = cootocsr(edge_index, edge_weight, v_num)

        print(f"[DEBUG] Data converted to sparse format")

        print(f"[DEBUG] Creating model: {args.model}")

        train_func = None
        test_func = None
        model = None
        optimizer = None

        if args.model == 'GCN':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.8):
                    super(Net, self).__init__()
                    self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256)
                    self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes)
                    self.dropout = dropout

                def execute(self):
                    x, csc, csr = data.x, data.csc, data.csr
                    x = nn.relu(self.conv1(x, csc, csr))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.conv2(x, csc, csr)
                    return x

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=0)

            def train():
                model.train()
                train_mask = data.train_mask
                if len(train_mask.shape) > 1:
                    train_mask = train_mask[0]
                
                pred = model()[train_mask]
                label = data.y[train_mask]
                loss = nn.cross_entropy_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits = model()
                accs = []
                
                masks = [data.train_mask, data.val_mask, data.test_mask]
                for mask in masks:            
                    current_mask = mask[0] if len(mask.shape) > 1 else mask
                    y_true = data.y[current_mask]
                    logits_masked = logits[current_mask]
                    
                    pred, _ = jt.argmax(logits_masked, dim=1)
                    acc = pred.equal(y_true).sum().item() / current_mask.sum().item()
                    accs.append(acc)
                
                return accs

        elif args.model == 'GAT':
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = GATConv(dataset.num_features, 128, e_num, cached=True, normalize=True)
                    self.conv2 = GATConv(128, dataset.num_classes, e_num, cached=True, normalize=True)

                def execute(self):
                    x, csc = data.x, data.csc
                    x = nn.relu(self.conv1(x, csc))
                    x = nn.dropout(x)
                    x = nn.relu(self.conv2(x, csc))
                    return nn.log_softmax(x, dim=1)

            model = Net()
            optimizer = nn.Adam([
                dict(params=model.conv1.parameters(), weight_decay=1e-4),
                dict(params=model.conv2.parameters(), weight_decay=1e-4)
            ], lr=5e-3)

            def train():
                model.train()
                pred = model()[data.train_mask]
                label = data.y[data.train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits, accs = model(), []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    y_ = data.y[mask]
                    logits_ = logits[mask]
                    pred, _ = jt.argmax(logits_, dim=1)
                    acc = pred.equal(y_).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs

        elif args.model == 'GraphSAGE':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.8):
                    super(Net, self).__init__()
                    self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=256, cached=True, root_weight=False)
                    self.conv2 = SAGEConv(in_channels=256, out_channels=dataset.num_classes, cached=True, root_weight=False)
                    self.dropout = dropout

                def execute(self):
                    x, edge_index = data.x, data.edge_index
                    x = nn.relu(self.conv1(x, edge_index))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.conv2(x, edge_index)
                    return nn.log_softmax(x, dim=1)

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4)

            def train():
                model.train()
                train_mask = data.train_mask
                if len(train_mask.shape) > 1:
                    train_mask = train_mask[0]
                
                pred = model()[train_mask]
                label = data.y[train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits = model()
                accs = []
                
                masks = [data.train_mask, data.val_mask, data.test_mask]
                for mask in masks:            
                    current_mask = mask[0] if len(mask.shape) > 1 else mask
                    y_true = data.y[current_mask]
                    logits_masked = logits[current_mask]
                    
                    pred, _ = jt.argmax(logits_masked, dim=1)
                    acc = pred.equal(y_true).sum().item() / current_mask.sum().item()
                    accs.append(acc)
                
                return accs

        elif args.model == 'ChebNet2':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.5):
                    super(Net, self).__init__()
                    self.lin1 = nn.Linear(dataset.num_features, 256)
                    self.lin2 = nn.Linear(256, dataset.num_classes)
                    self.prop = ChebNetII(K=3)
                    self.dropout = dropout

                def execute(self):
                    x = data.x
                    csc, csr = data.csc, data.csr
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = nn.relu(self.lin1(x))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.lin2(x)
                    x = self.prop(x, csc, csr)
                    return x

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.01)

            def train():
                model.train()
                train_mask = data.train_mask
                if len(train_mask.shape) > 1:
                    train_mask = train_mask[0]
                
                pred = model()[train_mask]
                label = data.y[train_mask]
                loss = nn.cross_entropy_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits = model()
                accs = []
                
                masks = [data.train_mask, data.val_mask, data.test_mask]
                for mask in masks:            
                    current_mask = mask[0] if len(mask.shape) > 1 else mask
                    y_true = data.y[current_mask]
                    logits_masked = logits[current_mask]
                    
                    pred, _ = jt.argmax(logits_masked, dim=1)
                    acc = pred.equal(y_true).sum().item() / current_mask.sum().item()
                    accs.append(acc)
                
                return accs

        elif args.model == 'SGC':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.8):
                    super(Net, self).__init__()
                    self.conv1 = SGConv(in_channels=dataset.num_features, out_channels=64, K=2)
                    self.conv2 = SGConv(in_channels=64, out_channels=dataset.num_classes, K=2)
                    self.dropout = dropout

                def execute(self):
                    x, csc, csr = data.x, data.csc, data.csr
                    x = nn.relu(self.conv1(x, csc, csr))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.conv2(x, csc, csr)
                    return nn.log_softmax(x, dim=1)

            model = Net(dataset)
            optimizer = nn.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

            def train():
                model.train()
                pred = model()[data.train_mask]
                label = data.y[data.train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits, accs = model(), []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    y_ = data.y[mask]
                    logits_ = logits[mask]
                    pred, _ = jt.argmax(logits_, dim=1)
                    acc = pred.equal(y_).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs

        elif args.model == 'APPNP':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.5):
                    super(Net, self).__init__()
                    hidden = 64
                    self.lin1 = nn.Linear(dataset.num_features, hidden)
                    self.lin2 = nn.Linear(hidden, dataset.num_classes)
                    self.prop = APPNP(K=10, alpha=0.1)
                    self.dropout = dropout

                def execute(self):
                    x, csc, csr = data.x, data.csc, data.csr
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = nn.relu(self.lin1(x))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.lin2(x)
                    x = self.prop(x, csc, csr)
                    return nn.log_softmax(x, dim=1)

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

            def train():
                model.train()
                pred = model()[data.train_mask]
                label = data.y[data.train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits, accs = model(), []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    y_ = data.y[mask] 
                    logits_ = logits[mask]
                    pred, _ = jt.argmax(logits_, dim=1)
                    acc = pred.equal(y_).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs

        elif args.model == 'GPRGNN':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.5):
                    super(Net, self).__init__()
                    hidden = 64
                    self.lin1 = nn.Linear(dataset.num_features, hidden)
                    self.lin2 = nn.Linear(hidden, dataset.num_classes)
                    self.prop = GPRGNN(K=10, alpha=0.2, Init="PPR")
                    self.dropout = dropout

                def execute(self):
                    x, csc, csr = data.x, data.csc, data.csr
                    x = nn.dropout(x, self.dropout)
                    x = nn.relu(self.lin1(x))
                    x = nn.dropout(x, self.dropout)
                    x = self.lin2(x)
                    x = self.prop(x, csc, csr)
                    return nn.log_softmax(x, dim=1)

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

            def train():
                model.train()
                pred = model()[data.train_mask]
                label = data.y[data.train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits, accs = model(), []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    y_ = data.y[mask] 
                    logits_ = logits[mask]
                    pred, _ = jt.argmax(logits_, dim=1)
                    acc = pred.equal(y_).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs

        elif args.model == 'BernNet':
            class Net(nn.Module):
                def __init__(self, dataset, dropout=0.5):
                    super(Net, self).__init__()
                    hidden = 64
                    self.lin1 = nn.Linear(dataset.num_features, hidden)
                    self.lin2 = nn.Linear(hidden, dataset.num_classes)
                    self.prop = BernNet(K=5)
                    self.dropout = dropout

                def execute(self):
                    x = data.x
                    csc1, csr1 = data.csc1, data.csr1
                    csc2, csr2 = data.csc2, data.csr2
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = nn.relu(self.lin1(x))
                    x = nn.dropout(x, self.dropout, is_train=self.training)
                    x = self.lin2(x)
                    x = self.prop(x, csc1, csr1, csc2, csr2)
                    return nn.log_softmax(x, dim=1)

            model = Net(dataset)
            optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

            def train():
                model.train()
                pred = model()[data.train_mask]
                label = data.y[data.train_mask]
                loss = nn.nll_loss(pred, label)
                optimizer.step(loss)
                return loss

            def test():
                model.eval()
                logits, accs = model(), []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    y_ = data.y[mask] 
                    logits_ = logits[mask]
                    pred, _ = jt.argmax(logits_, dim=1)
                    acc = pred.equal(y_).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs

        else:
            raise ValueError(f"Model {args.model} not supported")

        print(f"[DEBUG] Model created")

        print(f"[DEBUG] Starting training loop for {args.epochs} epochs")

        train()
        best_val_acc = test_acc = 0

        for epoch in range(1, args.epochs + 1):
            loss = train()
            train_acc, val_acc, tmp_test_acc = test()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            status['running'] = True
            status['epoch'] = epoch
            status['finished'] = False
            
            try:
                loss_val = float(loss.item())
            except:
                try:
                    loss_val = float(loss.numpy()[0])
                except:
                    loss_val = float(loss)
            
            status['loss'] = loss_val
            status['acc'] = test_acc
            status['history']['loss'].append(loss_val)
            status['history']['acc'].append(test_acc)
            
            update_status(status)

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc, status['loss']))

            jt.sync_all()
            jt.gc()

        status['running'] = False
        status['finished'] = True
        status['acc'] = test_acc
        update_status(status)
        print(f"[DEBUG] Training finished, best test acc: {test_acc}")

    except Exception as e:
        import traceback
        print(f"[ERROR] Training failed: {e}")
        print(f"[TRAIN] {traceback.format_exc()}")
        status['running'] = False
        status['finished'] = True
        status['error'] = str(e)
        update_status(status)

if __name__ == '__main__':
    main()
