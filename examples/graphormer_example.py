import jittor as jt  
from jittor import nn  
import jittor_geometric.transforms as T  
from jittor_geometric.datasets import MoleculeNet  
from jittor_geometric.dataloader import DataLoader  
from jittor_geometric.data import Data  
import numpy as np  
from tqdm import tqdm
import sys
# Import your converted Graphormer components  
from jittor_geometric.nn.models.graphormer import Graphormer

def global_mean_pool(x: jt.Var, batch: jt.Var, size: int = None) -> jt.Var:
    """
    Args:
        x: jt.Var of shape [N, F] - node features
        batch: jt.Var of shape [N] - batch assignment
        size: int, total number of graphs in the batch
    Returns:
        jt.Var of shape [B, F] - graph-level mean features
    """
    if size is None:
        size = int(batch.max().item()) + 1
    B, F = size, x.shape[1]

    sum_out = jt.zeros((B, F), dtype=x.dtype)
    count = jt.zeros((B, 1), dtype=x.dtype)

    for i in range(B):
        mask = (batch == i).unsqueeze(-1)  # [N, 1]
        masked_x = x * mask               # 只有当前 batch 的节点保留
        sum_out[i] = masked_x.sum(dim=0)  # 累加
        count[i] = mask.sum()             # 节点数量

    return sum_out / jt.maximum(count, jt.ones_like(count))

def train_test_split(indices, test_size=0.8, random_state=42):  
    """Simple train-test split implementation"""  
    np.random.seed(random_state)  
    shuffled = np.random.permutation(indices)  
    split_idx = int(len(indices) * test_size)  
    # Convert to regular Python integers  
    return shuffled[split_idx:].tolist(), shuffled[:split_idx].tolist()  
  
# Enable CUDA if available  
if jt.has_cuda:  
    jt.flags.use_cuda = 1  
  
# Load ESOL dataset  
dataset = MoleculeNet(root="./", name="ESOL")  
print(f"Dataset: {dataset}")  
  
# Create Graphormer model  
model = Graphormer(  
    num_layers=3,  
    input_node_dim=dataset.num_node_features,  
    node_dim=128,  
    input_edge_dim=dataset.num_edge_features,  
    edge_dim=128,  
    output_dim=1,  # ESOL has 1 target  
    n_heads=4,  
    ff_dim=256,  
    max_in_degree=5,  
    max_out_degree=5,  
    max_path_distance=5,  
)  
 
test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=0.8, random_state=42)  
 
train_dataset = [dataset[i] for i in train_ids]  
test_dataset = [dataset[i] for i in test_ids]
  
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  
  
# Setup optimizer and loss  
optimizer = jt.optim.Adam(model.parameters(), lr=3e-4)  
loss_function = nn.L1Loss()

# Training loop  
for epoch in range(10):
    model.train()
    batch_loss = 0.0  
      
    for batch in tqdm(train_loader):  

        y = batch.y  
        output = model(batch)  
          
        # Global mean pooling for graph-level prediction  
        if hasattr(batch, 'batch'):  
            output = global_mean_pool(output, batch.batch)  
        else:  
            output = output.mean(dim=0, keepdim=True)  
          
        loss = loss_function(output, y) # 这里在优化edge encoding
        try : 
            optimizer.step(loss)
        except Exception as e:
            print(type(loss), loss)
            print(e)
        #     sys.exit(0)

        batch_loss += loss.item()

        del loss, output, y
        jt.sync_all()
        # jt.display_memory_info()
        jt.gc()

    print("TRAIN_LOSS", batch_loss / len(train_ids))  
  
    # Evaluation  
    model.eval()  
    batch_loss = 0.0
      
    with jt.no_grad():  
        for batch in tqdm(test_loader):  
            y = batch.y  
            output = model(batch)  
            
            # Global mean pooling for graph-level prediction  
            if hasattr(batch, 'batch'):  
                output = global_mean_pool(output, batch.batch)  
            else:  
                output = output.mean(dim=0, keepdim=True)
              
            loss = loss_function(output, y)  
            batch_loss += loss.item()  
  
    print("EVAL LOSS", batch_loss / len(test_ids))
jt.save(model.state_dict(), "graphormer_esol.pkl")