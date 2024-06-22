import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='add_cpp',
    ext_modules=[CppExtension('add_cpp', ['add.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)