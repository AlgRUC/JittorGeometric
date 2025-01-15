import argparse
import os.path as osp

from tqdm import tqdm
import jittor as jt
from jittor_geometric.datasets import QM9
import jittor_geometric.transforms as T
from jittor_geometric.dataloader import DataLoader
from jittor import nn
from jittor_geometric.nn import SchNet

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# dataset = QM9(path)
qm9_dataset = QM9(path, transform=T.NormalizeFeatures())
print(len(qm9_dataset))
# random split train/val/test = 8/1/1
split_dict = qm9_dataset.get_idx_split()

# # dataloader
train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=8, shuffle=True)
valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=8, shuffle=False)
test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=8, shuffle=False)

def train(model, loader, optimizer, target):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc='Training'):
        optimizer.zero_grad()
        pred = model(data.z, data.pos, data.batch)
        loss = nn.MSELoss()(pred.view(-1), data.y[:, target])
        optimizer.step(loss)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)
def evaluate(model, loader, target):
    model.eval()
    maes = []
    for data in loader:
        with jt.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)
    mae = jt.cat(maes, dim=0)
    return mae.mean()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jt.flags.use_cuda = 1

for target in range(12):
    # model, datasets = SchNet.from_qm9_pretrained(path, qm9_dataset, target)
    # train_dataset, val_dataset, test_dataset = datasets

    model = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                  num_gaussians=50, cutoff=args.cutoff)
    optimizer = jt.optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练循环
    best_val_mae = float('inf')
    for epoch in range(30):  # 训练30个epoch
        loss = train(model, train_loader, optimizer, target)
        val_mae = evaluate(model, valid_loader, target)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}')

    # 测试过程保持不变
    # loader = DataLoader(test_dataset, batch_size=256)

    maes = []
    for data in tqdm(test_loader):
        # data = data.to(device)
        with jt.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)

    mae = jt.cat(maes, dim=0)

    # Report meV instead of eV.
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
