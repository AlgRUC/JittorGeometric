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
from jittor_geometric.nn import GCNConv
from jittor_geometric.datasets import Planetoid

Step 2: Load a Dataset
-----------------------
We will use the popular `Planetoid` dataset (e.g., Cora) for this example:

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Getting the first graph

Step 3: Define a Simple GCN Model
-----------------------------------
Now, let's define a basic Graph Convolutional Network (GCN) model:

class GCNModel(jt.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        
    def execute(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = jt.relu(x)
        x = self.conv2(x, edge_index)
        return x

Step 4: Training the Model
---------------------------
Let's train the model on the dataset:

# Initialize the model
model = GCNModel(dataset.num_features, dataset.num_classes)

# Set optimizer
optimizer = jt.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = jt.nn.cross_entropy_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch+1}: Loss = {loss.item()}')

Step 5: Evaluate the Model
---------------------------
After training, evaluate the model's performance:

model.eval()
out = model(data)
pred = jt.argmax(out, dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).float().sum()
accuracy = correct / data.test_mask.sum()
print(f'Accuracy: {accuracy.item() * 100:.2f}%')

Congratulations, you have successfully trained and tested a GNN model using `jittor_geometric`!

Next Steps
-----------
- Explore more datasets: `Planetoid`, `Cora`, `Citeseer`, etc.
- Try other graph neural network layers like `SAGEConv`, `GATConv`, etc.
- Check out the documentation for more advanced features.
