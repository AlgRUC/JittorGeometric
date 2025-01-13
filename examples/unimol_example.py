import jittor as jt
import numpy as np
from tqdm import tqdm
import logging
import sys
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import sys,os
import copy
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.nn.models.unimol import UniMolModel
from jittor_geometric.data.conformer import ConformerGen
from huggingface_hub import hf_hub_download
import pickle
from jittor_geometric.jitgeo_loader import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("mol_classification")


class MolDataset(jt.dataset.Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.download_data()
        self.conformer_gen = ConformerGen(remove_hs=True)
        self.data, self.label = self.process_data(self.data_dir)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # 处理单个索引
            return self.data[idx], self.label[idx]
        else:
            # 处理多个索引的情况
            return self.index_select(idx)

    def index_select(self, idx):
        # 创建一个新的数据集实例
        dataset = copy.copy(self)
        
        # 处理不同类型的索引
        if isinstance(idx, slice):
            dataset.data = self.data[idx]
            dataset.label = self.label[idx]
        elif isinstance(idx, (list, np.ndarray, jt.Var)):
            if isinstance(idx, jt.Var):
                idx = idx.tolist()
            dataset.data = [self.data[i] for i in idx]
            dataset.label = [self.label[i] for i in idx]
        else:
            raise IndexError(f'Invalid index type: {type(idx)}')
        
        return dataset

    def __len__(self):
        return len(self.data)
    
    def download_data(self):
        """
        #     ! IF YOU MEET NETWORK ERROR, PLEASE TRY TO RUN THE COMMAND BELOW:
        # `export HF_ENDPOINT=https://hf-mirror.com`,
        # TO USE THE MIRROR PROVIDED BY Hugging Face.
        """
        hf_hub_download(repo_id=f"TGB-Seq/bace", filename=f"bace.pkl", local_dir=self.data_dir, repo_type="dataset")

    def process_data(self, data_dir):
        """
        Preprocesses input data by either generating conformers from SMILES or loading from SDF files.

        :param smiles_list: List of SMILES strings to generate conformers for.
        :param sdf_paths: List of SDF file paths (or a single path) containing molecular conformers.
        :return: Processed molecular input for the model.
        """
        pickle_file = os.path.join(data_dir, "bace.pkl")
        with open(pickle_file, "rb") as f:
            raw_data = pickle.load(f)
        
        atoms_list = []
        coordinates_list = []
        label_list = []
        for item in raw_data:
            atoms_list.append(item['atoms'])
            coordinates_list.append(item['coordinates'])
            label_list.append(item['label'])
        # Handle atoms and coordinates input directly (e.g., from LMDB)
        if atoms_list and coordinates_list:
            inputs = self.conformer_gen.transform_raw(atoms_list, coordinates_list)

        return inputs, label_list

    def get_idx_split(self, frac_train: float = 0.8, frac_valid: float = 0.1, frac_test: float = 0.1, seed: int = 42):
            assert np.isclose(frac_train + frac_valid + frac_test, 1.0)
            if seed is not None:
                np.random.seed(seed)
            # random split
            num_data = len(self.data)
            shuffled_indices = np.random.permutation(num_data)

            train_cutoff = int(frac_train * num_data)
            valid_cutoff = int((frac_train + frac_valid) * num_data)

            train_idx = jt.array(shuffled_indices[:train_cutoff])
            valid_idx = jt.array(shuffled_indices[train_cutoff:valid_cutoff])
            test_idx = jt.array(shuffled_indices[valid_cutoff:])

            split_dict = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx
            }
            return split_dict

def compute_loss(net_output, targets, reduce=True):
    lprobs = jt.nn.log_softmax(net_output.float(), dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    targets = targets.view(-1)
    loss = jt.nn.nll_loss(
        lprobs,
        targets,
        reduction="sum" if reduce else "none",
    )
    return loss

def train_epoch(model, train_loader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(**data)
        # Calculate cross entropy loss
        loss = compute_loss(output, target)
        optimizer.step(loss)
        total_loss += loss.item()
        probs = jt.nn.softmax(output.float(), dim=-1).view(
                -1, output.size(-1)
        )
        pred = probs.argmax(dim=1)[0]
        correct += (pred == target).sum().item()
        total += target.numel()
        
        pbar.set_postfix({'loss': total_loss/(batch_idx+1), 
                         'acc': 100.*correct/total})
    
    return total_loss/len(train_loader), correct/total

def evaluate(model, data_loader, mode='val'):
    """Evaluate model on validation or test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with jt.no_grad():
        for data, target in data_loader:
            output = model(**data)
            loss = compute_loss(output, target)
        
            total_loss += loss.item()
            probs = jt.nn.softmax(output.float(), dim=-1).view(
                    -1, output.size(-1)
            )
            pred = probs.argmax(dim=1)[0]
            correct += (pred == target).sum().item()
            total += target.numel()
    
    acc = correct/total
    avg_loss = total_loss/len(data_loader)
    
    logger.info(f'{mode.capitalize()} set: Average loss: {avg_loss:.4f}, '
                f'Accuracy: {acc:.4f}')
    
    return avg_loss, acc

def main():

    dataset_name = 'bace'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/{}'.format(dataset_name))
    bace_dataset = MolDataset(path)
    # random split train/val/test = 8/1/1
    split_dict = bace_dataset.get_idx_split()
    
    # Initialize model
    model = UniMolModel(
        model_path=None,  # Train new model without pretrained weights
        output_dim=2  # Binary classification output dimension
    )
    
    # Create datasets and dataloaders
    train_loader = DataLoader(
        bace_dataset[split_dict["train"]], 
        batch_size=8, 
        shuffle=True,
        collate_fn=model.batch_collate_fn
    )
    valid_loader = DataLoader(
        bace_dataset[split_dict["valid"]], 
        batch_size=8, 
        shuffle=True,
        collate_fn=model.batch_collate_fn
    )
    test_loader = DataLoader(
        bace_dataset[split_dict["test"]], 
        batch_size=8, 
        shuffle=False,
        collate_fn=model.batch_collate_fn
    )
    
    # Optimizer
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 30
    best_val_acc = 0
    best_model_state = None
    patience = 5  # Early stopping patience
    no_improve = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        logger.info(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}')
        
        # Validate
        val_loss, val_acc = evaluate(model, valid_loader, mode='val')
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve = 0
            logger.info(f'New best validation accuracy: {val_acc:.4f}')
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Test best model
    logger.info("Testing best model...")
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, mode='test')
    logger.info(f'Final test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main() 