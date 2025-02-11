# Evaluation for COMET 

## Kernel Evaluation 

### Kernel Performance 

**Evaluation on small batch sizes:** 

<img src=figures\kernel-small-batches.jpg width=40% />

Compared with cuBLAS-FP16 and TRT-LLM-W4A16, our proposed achieves an average 1.50x and 1.26x speedup (batch size = 2); 1.51x and 1.24x speedup (batch size = 4); 1.43x and 1.25x speedup (batch size = 8), respectively. 

**Evaluation on large batch sizes:** 

<img src=figures\kernel-performance.jpg width=40% />

On average, COMET-W4Ax outperforms cuBLAS-FP16, TRT-LLM-W4A116 and TRT-LLM-W8A8 by 2.88x, 1.77x and 1.33x, respectively.

### Ablation Study 

**Ablation Study on W4Ax kernel optimization (normalized latency):**

<img src=figures\kernel-optimization-ablation.jpg width=40% />

Without software pipelining, weight interleaving, and fast conversion, the proposed COMET-W4Ax kernel experiences performance degradations of 1.69x, 1.27x, and 1.53x, respectively. The results are based on the GEMM kernels of LLaMA-3-8B and LLaMA-3-70B with various batch sizes. 

**Ablation Study on SM scheduling:**

<img src=figures\ablation-study.jpg width=40% />

## End-to-End Evaluation 

### Small Batches Evaluation 

**Evaluation on small batch sizes (batch size = 4):**

<img src=figures\e2e-batch-4.jpg width=40%>

We evaluate the LLM inference throughput at the batch size of 4. Compared to SOTA quantization strategy in TensorRT-LLM (TRT-LLM-W4A16), **COMET shows a 1.18x throughput improvement with small batch sizes.**

### End-to-End Performance 

**End-to-End Performance Comparison with TensorRT-LLM and QServe:** 

<img src=figures\e2e-evaluation.jpg width=40% />

We set the TRT-LLM-W4A16 as the baseline. According to our evaluation, COMET achieves 2.02× and 1.63× higher throughput on average for two different input/output se-
quence length settings, respectively. 

**We further compare the normalized throughput with the concurrent work, Qserve. The results demonstrate that COMET achieves up to 1.47x and an average of 1.18x higher throughput than QServe.**

### Ablation Study on End-to-End Performance 

<img src=figures\e2e-ablation.jpg width=40% />

Compared to TRT-LLM-W4A16, only quantizing activation with COMET gains an average of 1.38x throughput improvement, specifically achieving 1.80x speedup for LLaMA-2-7B. However, using KV cache quantization alone achieves only an average 1.17x throughput improvement over the SOTA TensorRT-LLM system and cannot support large models such as LLaMA-3-70B and Qwen2-72B.

### Same Batches Evaluation

<img src=figures\throughput-same-batch.jpg width=40% />
