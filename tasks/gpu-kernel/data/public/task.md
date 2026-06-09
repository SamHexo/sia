# TriMul — Triangular Matrix Multiplication Kernel Optimization

## Overview

Your task is to write an optimized implementation of the `custom_kernel` function — a core computational primitive from AlphaFold2's architecture used in the GPUMode TriMul competition. You will iteratively improve your kernel's runtime on a fixed H100 benchmark.

## Operation

The outgoing TriMul operator from AlphaFold-3. Given a packed `data` tuple, the operation applies learnable projections, gating, a batched triangular matrix multiplication, and a fused output projection:

```python
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def custom_kernel_reference(data: Tuple) -> torch.Tensor:
    """Unoptimized reference implementation — your starting baseline."""
    inp, mask, weights, cfg = data
    # inp     : Tensor[B, N, N, C]  float32 CUDA
    # mask    : Tensor[B, N, N]     float32 CUDA  (upper-triangular, 1 = keep)
    # weights : dict of named fp32 parameters (see below)
    # cfg     : {"dim": C, "hidden_dim": H, "nomask": bool}

    dim        = cfg["dim"]        # C — input/output channel count
    hidden_dim = cfg["hidden_dim"] # H — intermediate projection size
    nomask     = cfg.get("nomask", True)
    B, N, _, C = inp.shape

    # 1. Input LayerNorm (learned scale/bias)
    Z = F.layer_norm(inp, [C], weight=weights["norm.weight"], bias=weights["norm.bias"])

    # 2. Gated linear projections: (B, N, N, C) → (B, N, N, H)
    left  = (Z @ weights["left_proj.weight"].T)  * torch.sigmoid(Z @ weights["left_gate.weight"].T)
    right = (Z @ weights["right_proj.weight"].T) * torch.sigmoid(Z @ weights["right_gate.weight"].T)
    out_gate = torch.sigmoid(Z @ weights["out_gate.weight"].T)

    # 3. Optional triangular mask applied to both left and right
    left_bhnn  = left.permute(0, 3, 1, 2)   # (B, H, N, N)
    right_bhnn = right.permute(0, 3, 1, 2)  # (B, H, N, N)
    if not nomask and mask is not None:
        left_bhnn  = left_bhnn  * mask.unsqueeze(1)
        right_bhnn = right_bhnn * mask.unsqueeze(1)

    # 4. Batched GEMM: hidden[b,h,i,j] = Σ_k left[b,h,i,k] * right[b,h,j,k]
    left_mat  = left_bhnn.reshape(B * hidden_dim, N, N)
    right_mat = right_bhnn.reshape(B * hidden_dim, N, N).transpose(1, 2)
    hidden = torch.bmm(left_mat, right_mat).reshape(B, hidden_dim, N, N)

    # 5. Output LayerNorm + gating + linear projection → (B, N, N, C)
    hidden_flat = hidden.permute(0, 2, 3, 1).reshape(-1, hidden_dim)
    hidden_norm = F.layer_norm(hidden_flat, [hidden_dim],
                               weight=weights["to_out_norm.weight"],
                               bias=weights["to_out_norm.bias"])
    out = (hidden_norm * out_gate.reshape(-1, hidden_dim)) @ weights["to_out.weight"].T
    return out.reshape(B, N, N, C)
```

### `weights` dict keys

| Key | Shape | Description |
|-----|-------|-------------|
| `norm.weight` | `(C,)` | Input LayerNorm scale |
| `norm.bias` | `(C,)` | Input LayerNorm bias |
| `left_proj.weight` | `(H, C)` | Left projection |
| `left_gate.weight` | `(H, C)` | Left gate |
| `right_proj.weight` | `(H, C)` | Right projection |
| `right_gate.weight` | `(H, C)` | Right gate |
| `out_gate.weight` | `(H, C)` | Output gate |
| `to_out_norm.weight` | `(H,)` | Output LayerNorm scale |
| `to_out_norm.bias` | `(H,)` | Output LayerNorm bias |
| `to_out.weight` | `(C, H)` | Output linear projection |

## Function Signature

```python
from typing import Tuple, Dict
import torch

def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    """
    Optimized outgoing TriMul operator.

    Args:
        data: tuple (inp, mask, weights, cfg) where
            inp     : float32 CUDA tensor of shape (B, N, N, C)
            mask    : float32 CUDA tensor of shape (B, N, N)  — upper-triangular
            weights : dict of named fp32 parameter tensors (see task.md)
            cfg     : dict with keys 'dim' (C), 'hidden_dim' (H), 'nomask' (bool)
    Returns:
        float32 CUDA tensor of shape (B, N, N, C)
    """
```

## Evaluation

The function is benchmarked on shape N=256, C=128. Runtime is measured as the median over 50 trials (after 10 warmup iterations) with `torch.cuda.synchronize()` for accurate GPU timing.

**Scoring**:
```
speedup = reference_median_time / solution_median_time
score   = speedup   (higher is better, baseline = 1.0)
```

A solution running at the same speed as the reference scores 1.0. A 3× speedup scores 3.0.

## Rules

1. Write `solution.py` containing your `custom_kernel(data)` function
2. Evaluate using `python {dataset_dir}/evaluate.py solution.py`
3. After each evaluation, `results.json` is written to your working directory — do not write it yourself
4. At the end of your run, your working directory **must** contain `solution.py`
5. No side effects inside `custom_kernel`: no file I/O, no print statements, no global state mutation between calls
6. Output must match the reference numerically (atol=1e-2, rtol=1e-2 in float32)
7. The function must handle any `(B, N, N, C)` shape, not just the benchmark shape

## Available Libraries

`torch`, `triton`, `numpy`, `scipy`. A CUDA GPU is required.

## Optimization Strategies

The reference has several bottlenecks to attack:

1. **Excessive kernel launches**: LayerNorm, sigmoid, multiply, bmm, and mask each launch a separate kernel. Each launch costs ~5–10 µs of overhead on top of memory transfers.
2. **FP32 matmul**: The `torch.bmm` call does not exploit tensor cores. Converting to FP16 for the matmul alone can give 4–8× speedup on that step.
3. **Memory-bound elementwise ops**: The gating sequence (norm → sigmoid → multiply) reads and writes `Z_norm` three times when it could be done in one pass.
4. **Redundant computations**: `Z_norm` is computed once but `sigmoid(Z_norm)` and `sigmoid(-Z_norm)` are computed as separate ops.

**Quick wins to try first:**
- `torch.compile(custom_kernel_reference, mode="max-autotune")` — zero code change, often 2–3× speedup
- Manual FP16 cast before `bmm`: `a.half() @ b.half()` through `bmm`
- Fuse the gating in Triton: one kernel that reads `Z_norm` once and writes `a` and `b` simultaneously

**Deeper optimizations:**
- Fuse LayerNorm + sigmoid gating into a single Triton kernel (eliminates 4–5 kernel launches)
- Use cuBLAS GEMM directly in FP16 for the triangular matmul (delegate to tensor cores)
- Fuse the output LayerNorm + gating into one kernel
- Integrate the upper-triangular mask into the matmul kernel (apply it per-tile at zero extra cost)

## Profiling

```python
# Quick profiling with torch.profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    custom_kernel(data)  # data = (inp, mask, weights, cfg)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Dataset Directory Layout

```
{dataset_dir}/
├── evaluate.py     ← evaluation script
└── task.md         ← this file

{working_dir}/      ← your read/write workspace (initially empty)
```

Your shell's working directory is `{working_dir}`. Use absolute paths to access `{dataset_dir}`.

## Evaluation Script

```bash
python {dataset_dir}/evaluate.py solution.py
```

## Generalization

The development benchmark uses `B=1, N=256, C=H=128`. The private score averages over multiple shapes:
`N ∈ {128, 192, 256, 320, 384}` with `B=C=H=128`. Solutions that hardcode tile sizes or assume a specific N will generalize poorly.
