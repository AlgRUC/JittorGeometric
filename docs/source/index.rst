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

   import jittor as jt
   import jittor_geometric as jg
   
   # Load a dataset
   data = jg.datasets.Cora()
   
   # Create a GCN model
   model = jg.nn.models.GCN(
       in_channels=data.num_features,
       hidden_channels=64,
       out_channels=data.num_classes,
       num_layers=2
   )
   
   # Train the model
   optimizer = jt.optim.Adam(model.parameters())
   # ... training loop

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

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials
   :hidden:

   examples/README

Community & Support
-------------------

- **GitHub**: `JittorGeometric Repository <https://github.com/Jittor/JittorGeometric>`_
- **Issues**: Report bugs and request features
- **Discussions**: Join our community discussions
- **Citation**: How to cite JittorGeometric in your research

Changelog
---------

**v2.0.0** brings major improvements:

- 🆕 **New Models**: Added 15+ new GNN architectures
- 🚀 **Performance**: 2-3x speedup over v1.0
- 📚 **Documentation**: Comprehensive API documentation and tutorials
- 🔧 **API Changes**: Simplified and more intuitive API design
- 🌐 **Distributed**: Enhanced distributed training capabilities

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`