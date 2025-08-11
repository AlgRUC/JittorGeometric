import os.path as osp
import jittor as jt
from jittor import nn    
import jittor_geometric.transforms as T    
from jittor_geometric.datasets import QM9  
from jittor_geometric.dataloader import DataLoader    
from jittor_geometric.data import Data    
import numpy as np    
from tqdm import tqdm  
import sys  
  
# Import your TransformerM implementation  
from jittor_geometric.nn.models import TransformerM
  
def global_mean_pool(x: jt.Var, batch: jt.Var, size: int = None) -> jt.Var:  
    """Graph-level pooling for molecular property prediction"""  
    if size is None:  
        size = int(batch.max().item()) + 1  
    B, F = size, x.shape[1]  
  
    sum_out = jt.zeros((B, F), dtype=x.dtype)  
    count = jt.zeros((B, 1), dtype=x.dtype)  
  
    for i in range(B):  
        mask = (batch == i).unsqueeze(-1)  
        masked_x = x * mask  
        sum_out[i] = masked_x.sum(dim=0)  
        count[i] = mask.sum()  
  
    return sum_out / jt.maximum(count, jt.ones_like(count))  
  
def qm9_train_test_split(dataset_size, seed=42):  
    """QM9 standard split following Transformer-M paper"""  
    np.random.seed(seed)  
    indices = np.arange(dataset_size)  
      
    # QM9 standard split: ~110k train, ~10k valid, ~10k test  
    test_size = 10831  
    valid_size = 10000  
      
    # Shuffle indices  
    shuffled = np.random.permutation(indices)  
      
    # Split  
    test_idx = shuffled[:test_size]  
    valid_idx = shuffled[test_size:test_size + valid_size]  
    train_idx = shuffled[test_size + valid_size:]  
      
    return train_idx.tolist(), valid_idx.tolist(), test_idx.tolist()  
  
# Enable CUDA if available  
if jt.has_cuda:  
    jt.flags.use_cuda = 1  
  
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# dataset = QM9(path)
qm9_dataset = QM9(path, transform=T.NormalizeFeatures()) 
print(f"Dataset: {qm9_dataset}")  
print(f"Number of samples: {len(qm9_dataset)}")  
print(f"Node features: {qm9_dataset.num_node_features}")  
print(f"Edge features: {qm9_dataset.num_edge_features}")  
  
# QM9 target selection (0-18 available targets)  
target_idx = 7  # U0 (Internal energy at 0K) - commonly used target  
  
# Create train/valid/test splits  
split_dict = qm9_dataset.get_idx_split()

# # dataloader
train_loader = DataLoader(qm9_dataset[split_dict["train"]], batch_size=1, shuffle=True)
valid_loader = DataLoader(qm9_dataset[split_dict["valid"]], batch_size=1, shuffle=False)
test_loader = DataLoader(qm9_dataset[split_dict["test"]], batch_size=1, shuffle=False)

# train_ids = split_dict["train"]
# # Calculate normalization statistics from training set  
# train_targets = [qm9_dataset[i].y[target_idx].item() for i in train_ids]  
# train_mean = np.mean(train_targets)  
# train_std = np.std(train_targets)
  
print(f"Target {target_idx} statistics:")  
# print(f"Train mean: {train_mean:.4f}, std: {train_std:.4f}")  
  
# Create TransformerM model with QM9-specific configuration  
model = TransformerM(  
    num_layers=3,  # Following Transformer-M paper  
    input_node_dim=qm9_dataset.num_node_features,  
    node_dim=128,  # Hidden dimension  
    input_edge_dim=qm9_dataset.num_edge_features,  
    edge_dim=128,  
    output_dim=1,  # Single target prediction  
    n_heads=4,    # Multi-head attention  
    ff_dim=128,    # Feed-forward dimension  
    max_in_degree=5,  
    max_out_degree=5,  
    max_path_distance=5,  
    # Transformer-M specific parameters  
    add_3d=True,           # Enable 3D molecular bias  
    num_3d_bias_kernel=128, # Gaussian kernels for 3D encoding  
    no_2d=False,           # Keep 2D graph structure  
    mode_prob="0.2,0.2,0.6"  # Multi-modal training probabilities  
)  
  
# Setup optimizer and loss  
optimizer = jt.optim.Adam(model.parameters(), lr=2e-4)  
loss_function = nn.L1Loss()  # MAE loss commonly used for QM9  
  
# Training configuration  
num_epochs = 10
best_valid_loss = float('inf')  
  
print("Starting training...")  
  
for epoch in range(num_epochs):  
    # Training phase  
    model.train()  
    train_loss = 0.0  
    train_count = 0  
      
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):  
        # Extract target  
        y = batch.y[:, target_idx].unsqueeze(-1)  # Select specific target  
          
        # Normalize target  
        # y_norm = (y - train_mean) / train_std  
          
        # Forward pass  
        output = model(batch) 
        
          
        # Global mean pooling for graph-level prediction  
        if hasattr(batch, 'batch'):  
            output = global_mean_pool(output, batch.batch)  
        else:  
            output = output.mean(dim=0, keepdim=True)
          
        # Compute loss on normalized targets  
        loss = loss_function(output, y)  
          
        # Backward pass  
        optimizer.step(loss)  
          
        train_loss += loss.item()  
        train_count += 1  
          
        # Memory management  
        del loss, output, y 
        jt.sync_all()  
        jt.gc()  
      
    avg_train_loss = train_loss / train_count  
      
    # Validation phase  
    model.eval()  
    valid_loss = 0.0  
    valid_count = 0  
      
    with jt.no_grad():  
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Valid"):  
            y = batch.y[:, target_idx].unsqueeze(-1)  
              
            # Forward pass  
            output = model(batch)  
              
            # Global mean pooling  
            if hasattr(batch, 'batch'):  
                output = global_mean_pool(output, batch.batch)  
            else:  
                output = output.mean(dim=0, keepdim=True)  
              
            # Denormalize predictions for evaluation  
            # output_denorm = output * train_std + train_mean  
              
            # Compute loss on original scale  
            loss = loss_function(output, y)  
              
            valid_loss += loss.item()  
            valid_count += 1  
      
    avg_valid_loss = valid_loss / valid_count  
      
    print(f"Epoch {epoch+1}/{num_epochs}:")  
    print(f"  Train Loss: {avg_train_loss:.6f}")  
    print(f"  Valid Loss: {avg_valid_loss:.6f}")  
      
    # Save best model  
    if avg_valid_loss < best_valid_loss:  
        best_valid_loss = avg_valid_loss  
        jt.save(model.state_dict(), f"transformer_m_qm9_target_{target_idx}_best.pkl")  
        print(f"  New best model saved! Valid Loss: {best_valid_loss:.6f}")  
  
# Final test evaluation  
print("\nFinal test evaluation...")  
model.eval()  
test_loss = 0.0  
test_count = 0  
  
with jt.no_grad():  
    for batch in tqdm(test_loader, desc="Test"):  
        y = batch.y[:, target_idx].unsqueeze(-1)  
          
        output = model(batch)  
          
        if hasattr(batch, 'batch'):  
            output = global_mean_pool(output, batch.batch)  
        else:  
            output = output.mean(dim=0, keepdim=True)  
          
        # Denormalize predictions  
        # output_denorm = output * train_std + train_mean  
          
        loss = loss_function(output, y)  
        test_loss += loss.item()  
        test_count += 1  
  
avg_test_loss = test_loss / test_count  
print(f"Final Test Loss (MAE): {avg_test_loss:.6f}")  
  
# Save final model  
jt.save(model.state_dict(), f"transformer_m_qm9_target_{target_idx}_final.pkl")  
print("Training completed!")