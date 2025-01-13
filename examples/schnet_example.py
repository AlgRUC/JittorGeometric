import argparse
import os.path as osp

from tqdm import tqdm
import jittor as jt
from jittor_geometric.datasets import QM9
import jittor_geometric.transforms as T
from jittor_geometric.jitgeo_loader import DataLoader
from jittor_geometric.nn import SchNet

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# dataset = QM9(path)
qm9_dataset = QM9(path, transform=T.NormalizeFeatures())
# random split train/val/test = 8/1/1
# split_dict = qm9_dataset.get_idx_split()

# # dataloader
# train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=8, shuffle=True)
# valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=8, shuffle=False)
# test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=8, shuffle=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jt.flags.use_cuda = 1

for target in range(12):
    model, datasets = SchNet.from_qm9_pretrained(path, qm9_dataset, target)
    train_dataset, val_dataset, test_dataset = datasets

    # model = model.to(device)
    loader = DataLoader(test_dataset, batch_size=256)

    maes = []
    for data in tqdm(loader):
        # data = data.to(device)
        with jt.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)

    mae = jt.cat(maes, dim=0)

    # Report meV instead of eV.
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
