from rdkit import Chem
import jittor as jt
import numpy as np
from tqdm import tqdm
import logging
import sys
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import sys,os
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root)
from jittor_geometric.nn.models.unimol import UniMolModel
from jittor_geometric.data.conformer import ConformerGen

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("mol_classification")


class MolDataset(jt.dataset.Dataset):
    """Dataset class for molecular data"""
    def __init__(self, smiles_list, labels, conformer_gen):
        self.smiles_list = smiles_list
        self.labels = labels
        self.conformer_gen = conformer_gen
        
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        # Generate conformer data for a single molecule
        mol_data = self.conformer_gen.single_process(smiles)
        return mol_data, label

    def __len__(self):
        return len(self.smiles_list)

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
        loss = jt.nn.cross_entropy_loss(output, target)
        
        optimizer.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
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
            loss = jt.nn.cross_entropy_loss(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    
    acc = correct/total
    avg_loss = total_loss/len(data_loader)
    
    logger.info(f'{mode.capitalize()} set: Average loss: {avg_loss:.4f}, '
                f'Accuracy: {acc:.4f}')
    
    return avg_loss, acc

def main():
    # Example data - should be replaced with actual data
    smiles_list = ["C1=CC=CC=C1", "CCO", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * 25  
    labels = np.array([0, 1, 0, 1] * 25)  # Expanded dataset for demonstration
    
    # Dataset split (8:1:1)
    # First split out test set
    smiles_temp, smiles_test, y_temp, y_test = train_test_split(
        smiles_list, labels, test_size=0.1, random_state=42
    )
    # Then split validation set from remaining data
    smiles_train, smiles_val, y_train, y_val = train_test_split(
        smiles_temp, y_temp, test_size=0.111, random_state=42  # 0.111 ≈ 1/9
    )
    
    # Initialize conformer generator
    conformer_gen = ConformerGen(remove_hs=True)
    
    # Create datasets and dataloaders
    train_dataset = MolDataset(smiles_train, jt.array(y_train), conformer_gen)
    val_dataset = MolDataset(smiles_val, jt.array(y_val), conformer_gen)
    test_dataset = MolDataset(smiles_test, jt.array(y_test), conformer_gen)
    
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = jt.dataset.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = jt.dataset.DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = UniMolModel(
        model_path=None,  # Train new model without pretrained weights
        output_dim=2  # Binary classification output dimension
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
        val_loss, val_acc = evaluate(model, val_loader, mode='val')
        
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