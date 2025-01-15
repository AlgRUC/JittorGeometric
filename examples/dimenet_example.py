from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')  # type: ignore
import jittor as jt
import os.path as osp
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor import nn
from jittor_geometric.nn import DimeNet
from jittor_geometric.typing import Var
from jittor_geometric.datasets import QM9
import jittor_geometric.transforms as T
from jittor_geometric.dataloader import DataLoader
import jittor_geometric.dataloader
from tqdm import tqdm
import numpy as np

# Define MAE loss function
def mae_loss(pred: Var, target: Var) -> Var:
    return jt.abs(pred - target).mean()


# Run training
def train(model, loader, optimizer):
    model.train()
    loss_accum = 0

    train_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    # batch_data.z, batch_data.pos, batch_data.batch
    for step, batch_data in train_bar:
        pred = model(batch_data.z, batch_data.pos, batch_data.batch)
        loss = mae_loss(pred, batch_data.y)
        optimizer.step(loss)
        loss_accum += loss

        # 计算并显示当前平均loss  
        current_loss = loss_accum / (step + 1)  
        # 格式化显示，保留6位小数  
        train_bar.set_description(  
            f"Training (batch_loss={float(loss):.6f}, avg_loss={current_loss:.6f})"  
        )  

    return float(loss_accum / (step + 1))


def eval(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    # batch_data.z, batch_data.pos, batch_data.batch
    for step, batch_data in enumerate(tqdm(loader, desc="Iteration")):
        with jt.no_grad():
            pred = model(batch_data.z, batch_data.pos, batch_data.batch)
        y_true.append(batch_data.y)
        y_pred.append(pred)

    y_true = jt.cat(y_true, dim = 0)
    y_pred = jt.cat(y_pred, dim = 0)

    return float(mae_loss(y_pred, y_true))

def main():
    # data
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    qm9_dataset = QM9(path, transform=T.NormalizeFeatures())
    qm9_dataset.get(4)
    # random split train/val/test = 8/1/1
    split_dict = qm9_dataset.get_idx_split()

    # dataloader
    train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=8, shuffle=True)
    valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=8, shuffle=False)
    test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=8, shuffle=False)

    # model
    model = DimeNet(
            hidden_channels=128,
            out_channels=19,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

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
