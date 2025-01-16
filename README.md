# JittorGeometric

**[Documentation](https://algruc.github.io/JittorGeometric/index.html)** [![Documentation Status](https://readthedocs.org/projects/jittorgeometric/badge/?version=latest)](https://algruc.github.io/JittorGeometric/index.html)

JittorGeometric is a graph machine learning library based on the Jittor framework. As a Chinese-developed library, it is tailored for research and applications in Graph Neural Networks (GNNs), aiming to provide an efficient and flexible GNN implementation for researchers and engineers working with graph-structured data.

## Highlights
- **Easier Code Modification with JIT (Just-In-Time) Compilation**: JittorGeometric leverages JIT compilation to enable easier code modification without any pre-compilation requirements.
- **Optimized Sparse Matrix Computation**: JittorGeometric provides a rich set of operators and utilizes CuSparse to accelerate sparse matrix computations.
- **Comprehensive Domain Support**: Supports both basic and advanced GNNs, including spectral GNNs, dynamic GNNs, and molecular GNNs.

## Quick Tour

```python
### Dataset Selection
import os.path as osp
from jittor_geometric.datasets import Planetoid
import jittor_geometric.transforms as T

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

### Data Preprocess
from jittor_geometric.ops import cootocsr,cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
edge_index, edge_weight = data.edge_index, data.edge_attr
edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,v_num,
                        improved=False, add_self_loops=True)
with jt.no_grad():
   data.csc = cootocsc(edge_index, edge_weight, v_num)
   data.csr = cootocsr(edge_index, edge_weight, v_num)

### Model Definition
from jittor import nn
from jittor_geometric.nn import GCNConv

class GCN(nn.Module):
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

### Training
model = GCN(dataset)
optimizer = nn.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4) 
for epoch in range(200):
   model.train()
   pred = model()[data.train_mask]
   label = data.y[data.train_mask]
   loss = nn.nll_loss(pred, label)
   optimizer.step(loss)

```

## Supported Models
JittorGeometric includes implementations of popular GNN models, such as:

### Classic Graph Neural Networks

| Model      | Year | Venue  |
|------------|------|--------|
| [ChebNet](./examples/chebnet_example.py)    | 2016 | NeurIPS|
| [GCN](./examples/gcn_example.py)        | 2017 | ICLR   |
| [GraphSAGE](./examples/graphsage_example.py)  | 2017 | NeurIPS|
| [GAT](./examples/gat_example.py)        | 2018 | ICLR   |
| [SGC](./examples/sgc_example.py)        | 2019 | ICML   |
| [APPNP](./examples/appnp_example.py)      | 2019 | ICLR   |
| [GCNII](./examples/gcn2_example.py)      | 2020 | ICML   |

---

### Spectral Graph Neural Networks

| Model      | Year | Venue  |
|------------|------|--------|
| [GPRGNN](./examples/gprgnn_example.py)     | 2021 | ICLR   |
| [BernNet](./examples/bernnet_example.py)    | 2021 | NeurIPS|
| [ChebNetII](./examples/chebnet2_example.py)  | 2022 | NeurIPS|
| [EvenNet](./examples/evennet_example.py)    | 2022 | NeurIPS|
| [OptBasis](./examples/optbasis_example.py)   | 2023 | ICML   |

---

### Dynamic Graph Neural Networks

| Model      | Year | Venue  |
|------------|------|--------|
| [JODIE](./examples/jodie_example.py)      | 2019 | SIGKDD |
| [DyRep](./examples/dyrep_example.py)      | 2019 | ICLR   |
| [TGN](./examples/tgn_example.py)        | 2020 | ArXiv  |
| [GraphMixer](./examples/graphmixer_example.py) | 2022 | ICLR   |
| [Dygformer](./examples/dygformer_example.py)  | 2023 | NeurIPS|

---

### Molecular Graph Neural Networks

| Model      | Year | Venue  |
|------------|------|--------|
| [SchNet](./examples/schnet_example.py)     | 2017 | NeurIPS|
| [DimeNet](./examples/dimenet_example.py)    | 2020 | ICLR   |
| [EGNN](./examples/egnn_example.py)       | 2021 | ICML   |
| [SphereNet](./examples/spherenet_example.py)  | 2022 | ICLR   |
| [Uni-Mol](./examples/unimol_example.py)    | 2023 | ICLR   |

## Installation
Follow these steps to install JittorGeometric:

1. Install Jittor:
   ```bash
   python -m pip install git+https://github.com/Jittor/jittor.git
   ```
   or by following the [Jittor official documentation](https://cg.cs.tsinghua.edu.cn/jittor/).
2. Installing other dependencies:
   ```bash
   pip install astunparse==1.6.3 autograd==1.7.0 cupy==13.3.0 fairseq==0.12.2 Flask==3.1.0 jittor_offline==0.0.7 matplotlib==3.10.0 numpy==2.2.1 pandas==2.2.3 Pillow==11.0.0 pymetis==2023.1.1 pyparsing==3.2.0 pywebio==1.8.3 recommonmark==0.7.1 scikit_learn==1.6.0 scipy==1.14.1 setuptools==69.5.1 six==1.16.0 sphinx_rtd_theme==3.0.2 torch==2.5.1 torchvision==0.20.1 tqdm==4.66.4
   ```
3. Install the package:
   ```bash
   git clone https://github.com/lyyisbored/JittorGeometric.git
   cd "the project root directory that contains the setup.py file"
   pip install .
   ```
4. Verify the installation by running the gcn_example.py


## Contributors

This project is actively maintained by the **JittorGeometric Team**.

If you have any questions or would like to contribute, please feel free to contact [runlin_lei@ruc.edu.cn](mailto:runlin_lei@ruc.edu.cn).
