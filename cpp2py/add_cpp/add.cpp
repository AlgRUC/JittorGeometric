#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>


int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "A function which adds two numbers");
}