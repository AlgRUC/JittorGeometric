Installation
============

This guide will help you install `jittor_geometric`, a powerful library for graph neural networks.

Prerequisites
-------------
Before you start, ensure you have the following installed:

- Python 3.9+
- pip (Python package manager)
- Jittor (the underlying deep learning framework)

Install Jittor
--------------
First, you need to install Jittor. You can install it using the following command:

pip install jittor

For more details on Jittor installation, please visit the official Jittor installation guide: https://github.com/Jittor/jittor#installation

Install jittor_geometric
-------------------------
Once Jittor is installed, you can install `jittor_geometric` using pip. To install the latest release from PyPI, run:

pip install jittor-geometric

Alternatively, you can install the development version directly from GitHub by running:

pip install git+https://github.com/Jittor/jittor_geometric.git

System Requirements
-------------------
Ensure your system has the following dependencies installed:

- Linux, macOS, or Windows (with WSL2 for Windows)
- CUDA (for GPU support, optional)

For GPU support, ensure that the correct version of CUDA is installed, and your system supports CUDA-based operations.

Testing the Installation
-------------------------
Once the installation is complete, you can verify the installation by importing `jittor_geometric` in Python:

import jittor_geometric
print(jittor_geometric.__version__)

If no errors appear and the version prints correctly, the installation was successful.

Troubleshooting
---------------
If you encounter issues, please refer to the Jittor documentation for troubleshooting steps: https://github.com/Jittor/jittor
