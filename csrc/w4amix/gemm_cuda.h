#pragma once
#include <torch/extension.h>

namespace COMET::matmul {
void w4amix_gemm_forward_new(  torch::Tensor _in_feats, 
                                torch::Tensor _kernel, 
                                torch::Tensor _group,
                                torch::Tensor _wscales, 
                                torch::Tensor _ascales, 
                                torch::Tensor _out_feats);

}
