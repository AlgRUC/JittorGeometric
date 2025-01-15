Get Started
===========

Welcome to `jittor_geometric`! This guide will walk you through the basic steps to get started with graph neural networks using this library.

Quick Start
-----------

Let's start by building a simple Graph Neural Network (GNN) model using `jittor_geometric`.

Step 1: Import Libraries
-------------------------
First, import the necessary libraries:

import jittor as jt
from jittor import nn
from jittor_geometric.nn import GCNConv
from jittor_geometric.datasets import Planetoid

Step 2: Load a Dataset
-----------------------
We will use the popular `Planetoid` dataset (e.g., Cora) for this example:

dataset = Planetoid(root='your_path', name='Cora')
data = dataset[0]  # Getting the first graph

# Prepare data
from jittor_geometric.ops import cootocsr,cootocsc

edge_index, edge_weight = data.edge_index, data.edge_attr 
edge_index, edge_weight = gcn_norm(edge_index, edge_weight,v_num, improved=False, add_self_loops=True)
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

Step 3: Define a Simple GCN Model
-----------------------------------
Now, let's define a basic Graph Convolutional Network (GCN) model:

class GCNModel(jt.Module):
    def __init__(self, dataset, dropout=0.8):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256,spmm=args.spmm)
        self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes,spmm=args.spmm)
        self.dropout = dropout

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return nn.log_softmax(x, dim=1)

Step 4: Training the Model
---------------------------
Let's train the model on the dataset:

# Initialize the model
model = GCNModel(dataset)

# Set optimizer
optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4) 

# Training loop
for epoch in range(200):
    model.train()
    pred = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)

Step 5: Evaluate the Model
---------------------------
After training, evaluate the model's performance:

model.eval()
out = model()
pred, _ = jt.argmax(out, dim=1)
y_test = data.y[data.test_mask]
accuracy = pred.equal(y_test).sum().item() / data.test_mask.sum().item()
print(f'Accuracy: {accuracy.item() * 100:.2f}%')

Congratulations, you have successfully trained and tested a GNN model using `jittor_geometric`!

Next Steps
-----------
- Explore more datasets: `Planetoid`, `Cora`, `Citeseer`, etc.
- Try other graph neural network layers like `SAGEConv`, `GATConv`, etc.
- Check out the documentation for more advanced features.
