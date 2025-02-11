# COMET: Towards Practical W4A4KV4 LLMs Serving 

---
## Installtion

1. setup tutorial
```bash=
conda create -n comet python=3.10 -y
conda activate comet
pip install --upgrade pip  # enable PEP 660 support
# This is optional if you prefer to use built-in nvcc
conda install -c nvidia cuda-toolkit -y
pip install torch==2.4.0
```

Note that COMET-W4Ax kernel relies on cutlass, please follow the instructions in [CUTLASS v3.1.0](https://github.com/NVIDIA/cutlass/tree/v3.1.0) to install the corresponding dependencies.

2. Compile the CUDA kernels.

Please return to the comet directory and execute the following commands:

```bash=
TORCH_CUDA_ARCH_LIST=8.0 python setup.py install
``` 

## Usage with COMET-W4Ax kernel

We have already integrated the COMET-W4Ax kernel into the provided python package, COMET. Users can replace the pytorch NN module with COMET to achieve the corresponding mixed-precision GEMM computing.

```python 
class LlamaMLP(nn.Module):
# original code 
 def __init__(self, args, group_size: int) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size

        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False, group_size=group_size
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, group_size=group_size
        )

# code for comet (replacing the given codes as follows)
from comet import W4AxLinear
class COMETLlamaMLP(nn.Module):
 def __init__(self, args, group_size: int) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size

        self.gate_up_proj = W4AxLinear(
            hidden_size, 2 * intermediate_size, bias=False, group_size=group_size
        )
        self.down_proj = W4AxLinear(
            intermediate_size, hidden_size, bias=False, group_size=group_size
        )
```

## Performance Evaluation 

### Kernel Performance 

**Evaluation on large batch sizes:** 

<img src=figures\kernel-performance.jpg width=40% />

On average, COMET-W4Ax outperforms cuBLAS-FP16, TRT-LLM-W4A116 and TRT-LLM-W8A8 by 2.88x, 1.77x and 1.33x, respectively.

## End-to-End Evaluation 

**End-to-End Performance Comparison with TensorRT-LLM and QServe:** 

<img src=figures\e2e-evaluation.jpg width=40% />

We set the TRT-LLM-W4A16 as the baseline. According to our evaluation, COMET achieves 2.02× and 1.63× higher throughput on average for two different input/output se-
quence length settings, respectively. 

**We further compare the normalized throughput with the concurrent work, Qserve. The results demonstrate that COMET achieves up to 1.47x and an average of 1.18x higher throughput than QServe.**

For more evaluation results, please refer [Evaluation Results](./Evaluation.md)
