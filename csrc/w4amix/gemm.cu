#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm_cuda.h"

namespace COMET::matmul {

  void buildSubmodule(py::module &mod) {
    py::module m = mod.def_submodule("w4Ax", "Matmul Functions");
    m.def("W4AxLinear", &w4amix_gemm_forward_new, "our w4amix gemm kernel");

  }
  
}