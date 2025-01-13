# Jittor Geometric

## Introduction
Jittor Geometric is a graph machine learning library based on the Jittor framework. As a Chinese-developed library, it is tailored for research and applications in Graph Neural Networks (GNNs), aiming to provide an efficient and flexible GNN implementation for researchers and engineers working with graph-structured data.

## Highlights
- **Easier Code Modification with JIT (Just-In-Time) Compilation**: Jittor Geometric leverages JIT compilation to enable easier code modification without any pre-compilation requirements.
- **Optimized Sparse Matrix Computation**: Jittor Geometric provides a rich set of operators and utilizes CuSparse to accelerate sparse matrix computations.
- **Comprehensive Spectral Domain Support**: Supports various spectral methods, enabling a wide range of spectral-based GNN architectures.
- **Rich Dynamic Dataset Support**: Easily handle dynamic datasets, allowing for efficient processing and transformation of graph data.

## Quick Tour for New Users

### Dataset Selection
Jittor Geometric supports a variety of graph datasets, including Cora, CiteSeer, and PubMed. Here’s an example of loading the Cora dataset:

```python
import os.path as osp
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
```

### Model Definition
The following code defines a simple two-layer Graph Convolutional Network (GCN):

```python
from jittor import nn
from jittor_geometric.nn import GCNConv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def execute(self):
        x, edge_index = data.x, data.edge_index
        x = nn.relu(self.conv1(x, edge_index))
        x = nn.dropout(x)
        x = self.conv2(x, edge_index)
        return nn.log_softmax(x, dim=1)
```

## Implemented GNN Models
Jittor Geometric includes implementations of popular GNN models, such as:

- **Graph Convolutional Network (GCN)**
- **Chebyshev Network (ChebNet)**
- **Simplified Graph Convolution (SGC)**
- **GCNII (Graph Convolutional Network II)**

## Installation
Follow these steps to install Jittor Geometric:

1. Install the Jittor framework by following the [Jittor official documentation](https://cg.cs.tsinghua.edu.cn/jittor/).
2. Clone this repository and navigate to the project directory:
   ```bash
   git clone <repo_url>
   cd jittor_geometric
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Jittor Geometric from the source code:
   ```bash
   git clone https://github.com/AlgRUC/JittorGeometric.git
   cd "the project root directory that contains the setup.py file"
   pip install .