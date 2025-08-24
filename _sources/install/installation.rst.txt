============
Installation
============

This document provides the installation instructions for jittor_geometric

System Requirements
-------------------

- **Python**: 3.7 or higher
- **GCC** (Linux only): 5.4 or higher
- **CMake**: 3.10 or higher

GPU Environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **CUDA**: 11.2 (for NVIDIA GPU support)
- **cuDNN**: Compatible with the installed CUDA version

Ascend Environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **CANN (Compute Architecture for Neural Networks) Toolkit**: 6.0.1 or higher
- **Ascend Driver**: Compatible with your hardware (e.g., Ascend 910B/910B1/910B2)
- **ACL Python API**: Installed as part of the CANN Toolkit
- **AscendC Compiler**: Included in the CANN Toolkit

Before running on Ascend, configure the environment variables::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh

Also ensure to preload the correct OpenMP library (adjust path to your environment)::

    export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1


Dependencies
-----------------

- ase==3.24.0
- astunparse==1.6.3
- autograd==1.7.0
- cupy==13.3.0
- Flask==3.1.0
- einops
- huggingface_hub==0.27.1
- jittor==1.3.9.14
- numpy==1.24.0
- pandas==2.2.3
- Pillow==11.1.0
- PyMetis==2023.1.1
- pyparsing==3.2.1
- pywebio==1.8.3
- recommonmark==0.7.1
- schnetpack==2.0.0
- scikit_learn==1.6.1
- scipy==1.15.1
- setuptools==69.5.1
- six==1.16.0
- sphinx_rtd_theme==3.0.2
- sympy==1.13.3
- tqdm==4.66.4

Installation Steps
------------------

1. Install Jittor::

    python -m pip install git+https://github.com/Jittor/jittor.git

2. Installing other dependencies, such as::

    pip install astunparse==1.6.3 autograd==1.7.0 cupy==13.3.0 numpy==1.24.0 pandas==2.2.3 Pillow==11.1.0 PyMetis==2023.1.1 six==1.16.0 pyparsing==3.2.1 scipy==1.15.1 setuptools==69.5.1 sympy==1.13.3 tqdm==4.66.4 einops huggingface_hub==0.27.1

3. Install the package::

    git clone https://github.com/AlgRUC/JittorGeometric.git
    cd JittorGeometric
    pip install .

4. Verify the installation
      Run the gcn_example.py to check if jittor_geometric is installed correctly


Troubleshooting
---------------

- Higher versions of cuda may not have been adapted, version 11.2 is recommended.
- On Linux, ensure that GCC 5.4 or higher is installed.
- Ensure that CMake 3.10 or higher is installed and accessible in your environment.

If you have any questions or would like to contribute, please feel free to contact runlin_lei@ruc.edu.cn.
