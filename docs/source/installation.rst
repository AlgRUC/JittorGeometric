============
Installation
============

This document provides the installation instructions for jittor_geometric

System Requirements
-------------------

- **Python**: 3.7 or higher.
- **GCC** (Linux only): 5.4 or higher.
- **CMake**: 3.10 or higher.
- **CUDA** (optional): 11.2 (for GPU support).
- **cuDNN** (optional): Compatible with the installed CUDA version.

Core Dependencies
-----------------

- jittor==1.3.9.14
- numpy
- tqdm
- pybind11
- psutil
- pillow
- scipy
- requests
- pandas
- pyparsing
- scikit-learn
- six


Installation Steps
------------------

1. Install Jittor::

    python -m pip install git+https://github.com/Jittor/jittor.git

2. Installing dependencies

3. Install the package::

    git clone https://github.com/AlgRUC/JittorGeometric.git
    cd "the project root directory that contains the setup.py file"
    pip install .

4. Verify the installation
      Run the gcn_example.py to check if jittor_geometric is installed correctly


Troubleshooting
---------------

- Higher versions of cuda may not have been adapted, version 11.2 is recommended.
- On Linux, ensure that GCC 5.4 or higher is installed.
- Ensure that CMake 3.10 or higher is installed and accessible in your environment.

For more bugs you can contact us at the project homepage https://github.com/AlgRUC/JittorGeometric