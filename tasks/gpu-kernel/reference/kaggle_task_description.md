# GPUMode TriMul — Triangular Matrix Multiplication Competition

## Overview

GPUMode is an open community for GPU kernel development that hosts competitions for domain experts. The **TriMul** competition asks participants to write the fastest possible implementation of a triangular matrix multiplication primitive — a core building block in AlphaFold2's architecture for protein structure prediction.

Each GPU architecture (NVIDIA H100, A100, B200, AMD MI300X) has its own leaderboard, since performant implementations differ across hardware. Submissions must pass correctness checks before runtime is measured.

Your submission is a single Python file containing a `custom_kernel(data)` function. You develop and profile it against an H100 benchmark. The evaluation runs on a fixed set of input shapes and reports the median runtime.

## The Operation

The outgoing TriMul operator from AlphaFold-3, applied to batched pair representations. `custom_kernel` receives a single `data` tuple:

```
data = (inp, mask, weights, cfg)

  inp     : Tensor[B, N, N, C]   float32 CUDA  — pair representation
  mask    : Tensor[B, N, N]      float32 CUDA  — upper-triangular (1 = keep)
  weights : dict of named fp32 parameter tensors
  cfg     : {"dim": C, "hidden_dim": H, "nomask": bool}
```

The algorithm:

```
Z        = LayerNorm(inp, w=norm.weight, b=norm.bias)    # (B,N,N,C)
left     = (Z @ left_proj.T)  ⊙ σ(Z @ left_gate.T)     # (B,N,N,H)  gated projection
right    = (Z @ right_proj.T) ⊙ σ(Z @ right_gate.T)    # (B,N,N,H)
out_gate = σ(Z @ out_gate.T)                             # (B,N,N,H)

left  ⊙= mask   (broadcast over H)   # upper-tri mask on i,k plane
right ⊙= mask                         # upper-tri mask on j,k plane

hidden[b,h,i,j] = Σ_k left[b,h,i,k] · right[b,h,j,k]  # batched GEMM

output = (LayerNorm(hidden, H) ⊙ out_gate) @ to_out.T   # (B,N,N,C)
```

The operation is dominated by two bottlenecks:
- **Memory-bound elementwise ops**: LN, sigmoid, and gating each touch the full `(B, N, N, H)` tensor
- **Compute-bound GEMM**: O(B·H·N²) with large N, amenable to tensor core acceleration in FP16

## Benchmark

**Primary benchmark shape**: `N=256, C=128` (matching typical AlphaFold2 inference at medium sequence length)

**Hardware**: NVIDIA H100 SXM5 (80 GB)

**Measurement**: median over 50 trials with 10 warmup iterations, `torch.cuda.synchronize()` for accurate GPU timing

**Correctness tolerance**: atol=1e-2, rtol=1e-2 (float32 output)

## Known Results (H100)

From the TTT-Discover paper (reported runtimes in µs):

| Method | H100 (µs) |
|--------|-----------|
| 5th human | 4,233 |
| 4th human | 3,655 |
| 3rd human | 2,546 |
| 2nd human | 2,368 |
| **1st human** | **1,371** |
| Best-of-25600 (gpt-oss-120b) | 5,390 |
| **TTT-Discover (gpt-oss-120b)** | **1,161** |

TTT-Discover achieves **~1.18× speedup** over the best human submission and **~15%+ improvement** over all human submissions. The key insight found by the agent: the operation is **memory-bound** because of the surrounding elementwise ops, so the winning strategy is maximum **operation fusion** to reduce memory traffic and kernel launch overhead.

## Winning Strategy (Expert Review, GPUMode Organizers)

> *"The referenced solution correctly determined that the problem is memory bound because of the surrounding point-wise operations so the agent focuses as much as possible on operation fusions, lowering the memory traffic and kernel launch overhead."*
>
> *"Its strategy is to reduce memory bandwidth via fusions, lower precision and delegating the big matrix multiplications to cuBLAS, as those are non-trivial to beat. This is similar to the current best human solutions, but executed on better."*
> — Matej Sirovatka, Alex Zhang, Mark Saroufim (GPUMode)

Concretely, the winning kernel:
1. **Fuses** the input LayerNorm operations into a single kernel
2. **Fuses** sigmoid + elementwise multiplication (input gating)
3. **Fuses** the output LayerNorm + gating
4. **Converts inputs to FP16** and delegates the triangular matmul to **cuBLAS/cuBLASLt**, leveraging tensor cores

## Starting Point

The unoptimized PyTorch reference (equivalent to a naive first submission):

```python
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def custom_kernel(data: Tuple) -> torch.Tensor:
    inp, mask, weights, cfg = data
    dim, H = cfg["dim"], cfg["hidden_dim"]
    nomask = cfg.get("nomask", True)
    B, N, _, C = inp.shape

    Z = F.layer_norm(inp, [C], weight=weights["norm.weight"], bias=weights["norm.bias"])

    left  = (Z @ weights["left_proj.weight"].T)  * torch.sigmoid(Z @ weights["left_gate.weight"].T)
    right = (Z @ weights["right_proj.weight"].T) * torch.sigmoid(Z @ weights["right_gate.weight"].T)
    out_gate = torch.sigmoid(Z @ weights["out_gate.weight"].T)

    left_bhnn  = left.permute(0, 3, 1, 2)
    right_bhnn = right.permute(0, 3, 1, 2)
    if not nomask and mask is not None:
        left_bhnn  = left_bhnn  * mask.unsqueeze(1)
        right_bhnn = right_bhnn * mask.unsqueeze(1)

    left_mat  = left_bhnn.reshape(B * H, N, N)
    right_mat = right_bhnn.reshape(B * H, N, N).transpose(1, 2)
    hidden = torch.bmm(left_mat, right_mat).reshape(B, H, N, N)

    hidden_flat = hidden.permute(0, 2, 3, 1).reshape(-1, H)
    hidden_norm = F.layer_norm(hidden_flat, [H],
                               weight=weights["to_out_norm.weight"],
                               bias=weights["to_out_norm.bias"])
    out = (hidden_norm * out_gate.reshape(-1, H)) @ weights["to_out.weight"].T
    return out.reshape(B, N, N, C)
```

This baseline launches many separate CUDA kernels and runs the GEMM in FP32, leaving substantial room for improvement.

**Your goal is to beat the best human submission (1,371 µs on H100).**  
Scores above 2× speedup over the reference baseline are strong results. The TTT-Discover kernel achieves ~3.6× speedup over the unoptimized baseline.

## Available Libraries

`torch` · `triton` · `numpy`

Triton is pre-installed and can be used for custom fused kernels. For the triangular matmul, delegating to `torch.mm` / cuBLAS in FP16 is recommended over a hand-written Triton kernel, since cuBLAS already saturates tensor core utilization.

## Generalization

The development benchmark uses `N=256, C=128`. The private evaluation averages the geometric mean of runtimes over `N ∈ {128, 192, 256, 320, 384}` with `C=128` — consistent with the GPUMode leaderboard methodology of benchmarking across a fixed set of input shapes. A solution that hardcodes tile sizes or assumes a specific N will degrade on other shapes.
