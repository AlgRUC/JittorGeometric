JittorGeometric 2.0 Documentation
===========================================

.. image:: ../../assets/JittorGeometric_logo.png
   :align: center
   :width: 400px
   :alt: JittorGeometric Logo

Welcome to JittorGeometric
---------------------------

**JittorGeometric 2.0** is a comprehensive graph machine learning library built on the `Jittor <https://cg.cs.tsinghua.edu.cn/jittor/>`_ framework. As a Chinese-developed deep learning library, JittorGeometric provides state-of-the-art Graph Neural Network (GNN) implementations with enhanced performance and flexibility.

.. note::
   JittorGeometric 2.0 introduces significant enhancements including distributed training, dynamic graph support, mini-batch processing, and extended model architectures.

Key Features
------------

* **Just-In-Time (JIT) Compilation**: Easier code modification without pre-compilation requirements
* **Optimized Sparse Operations**: High-performance sparse matrix computations with CuSparse acceleration
* **Comprehensive Model Support**: Classic, spectral, dynamic, molecular, and transformer-based GNNs
* **Distributed Training**: Multi-GPU and multi-node training capabilities
* **Dynamic Graph Processing**: Parallel event-based dynamic graph support
* **Rich Dataset Collection**: Built-in support for popular graph datasets

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install/installation
   get_started/introduction

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   modules/nn
   data/data
   datasets/datasets
   dataloader/dataloader

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   partition/partition
   example/dist_gcn
   transforms/transforms
   ops/ops

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   io/io
   utils/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials

   examples/README

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`