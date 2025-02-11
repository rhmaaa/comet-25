#include <pybind11/pybind11.h>

// #include "asymmetric/asymmetric.h"
#include "w4amix/gemm.cu"
// #include "symmetric/symmetric.h"

PYBIND11_MODULE(_C, mod) {
//   COMET::matmul::buildSubmodule(mod);
  COMET::matmul::buildSubmodule(mod);
}
