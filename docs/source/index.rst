Welcome to JittorGeometric v2.0 Documentation
=============================================

.. image:: ../../assets/JittorGeometric_logo.png
   :width: 400
   :align: center

Overview
--------

**JittorGeometric 2.0** is a comprehensive library for machine learning on graph data, built on the powerful `Jittor <https://github.com/Jittor/jittor>`_ deep learning framework. 

This major version introduces significant improvements in performance, usability, and feature coverage, making it the go-to library for graph neural networks, temporal graph learning, and large-scale graph processing.

Key Features
------------

🚀 **High Performance**: Optimized CUDA kernels and distributed training support

🔬 **Rich Model Zoo**: 50+ state-of-the-art GNN models and layers

📊 **Comprehensive Datasets**: Built-in loaders for popular graph benchmarks

⚡ **Temporal Graphs**: Advanced support for dynamic and temporal graph learning

🧪 **Molecular AI**: Specialized modules for molecular property prediction

🌐 **Distributed Computing**: Seamless multi-GPU and multi-node training

Quick Start
-----------

.. code-block:: python

   ### Dataset Selection
   import os.path as osp
   from jittor_geometric.datasets import Planetoid
   import jittor_geometric.transforms as T
   import jittor as jt

   dataset = 'cora'
   path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
   dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
   data = dataset[0]
   v_num = data.x.shape[0]

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
         super(GCN, self).__init__()
         self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=256)
         self.conv2 = GCNConv(in_channels=256, out_channels=dataset.num_classes)
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

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   install/installation
   get_started/introduction

.. toctree::
   :maxdepth: 2
   :caption: Core Modules
   :hidden:

   modules/nn
   data/data
   datasets/datasets
   dataloader/dataloader

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features
   :hidden:

   partition/partition
   transforms/transforms
   ops/ops
   example/dist_gcn

.. toctree::
   :maxdepth: 2
   :caption: Utilities & I/O
   :hidden:

   io/io
   utils/utils


Community & Support
-------------------

- **GitHub**: `JittorGeometric Repository <https://github.com/Jittor/JittorGeometric>`_
- **Issues**: Report bugs and request features
- **Discussions**: Join our community discussions

Changelog
---------

**v2.0.0** brings major improvements:

- 🆕 **New Models**: Added 15+ new GNN architectures
- 🚀 **Performance**: Speedup over v1.0 in dynamic graphs
- 🔧 **NPU Implementation**: Allow JittorGeoemtric to run on NPU
- 🌐 **Distributed**: Enhanced distributed training capabilities

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`