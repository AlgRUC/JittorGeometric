import jittor as jt
import os.path as osp
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor import nn
from jittor_geometric.nn import EGNNConv, global_add_pool
from jittor_geometric.nn.conv.spherenet_conv import SphereNet
from jittor_geometric.typing import Var
from jittor_geometric.datasets import QM9
import jittor_geometric.transforms as T
from jittor_geometric.jitgeo_loader import DataLoader
import jittor_geometric.jitgeo_loader
from tqdm import tqdm
import numpy as np

# sample synthetic data (e.g., random graph)
def generate_data(num_nodes, num_edges):
    x = jt.randn((num_nodes, 6))  # 3 coordinates + 3 features
    edge_index = jt.randint(0, num_nodes, (2, num_edges))  # Random edge indices
    edge_attr = jt.randn((num_edges, 3))  # Random edge attributes
    return x, edge_index, edge_attr


# Define MAE loss function
def mae_loss(pred: Var, target: Var) -> Var:
    return jt.abs(pred - target).mean()


# Run training
def train(model, loader, optimizer):
    model.train()
    loss_accum = 0

    # batch_data.z, batch_data.pos, batch_data.pos
    for step, batch_data in enumerate(tqdm(loader, desc="Iteration")):
        pred = model(batch_data)
        loss = mae_loss(pred, batch_data.y)
        optimizer.step(loss)
        loss_accum += loss

    return float(loss_accum / (step + 1))


def eval(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with jt.no_grad():
        # batch_data.z, batch_data.pos, batch_data.pos
        for step, batch_data in enumerate(tqdm(loader, desc="Iteration")):
            pred = model(batch_data)
            y_true.append(batch_data.y)
            y_pred.append(pred)

        y_true = jt.cat(y_true, dim = 0)
        y_pred = jt.cat(y_pred, dim = 0)

        return float(mae_loss(y_pred, y_true))

def main():
    # data
    dataset_name = 'qm9'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/QM9')
    qm9_dataset = QM9(path, transform=T.NormalizeFeatures())
    # random split train/val/test = 8/1/1
    split_dict = qm9_dataset.get_idx_split()

    # dataloader
    train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=8, shuffle=True)
    valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=8, shuffle=False)
    test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=8, shuffle=False)

    # model
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        output_init='GlorotOrthogonal', use_node_features=True)

    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    best_valid_mae = 1000

    for epoch in range(1, 3):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train_mae = train(model, train_loader, optimizer)

            print('Evaluating...')
            valid_mae = eval(model, valid_loader)

            print('Testing...')
            test_mae = eval(model, test_loader)

            print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
            print(f'Best validation MAE so far: {best_valid_mae}')


if __name__ == "__main__":
    main()
