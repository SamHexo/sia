#!/usr/bin/env python3
"""
Evaluate a custom_kernel solution against the TriMul benchmark.

Usage:
    python evaluate.py solution.py

The solution.py must define a top-level `custom_kernel(data)` function where
data = (inp, mask, weights, cfg):
  - inp     : Tensor[B, N, N, C]  float32 CUDA
  - mask    : Tensor[B, N, N]     float32 CUDA  (upper-triangular)
  - weights : dict of named fp32 parameters
  - cfg     : dict with keys 'dim' (C), 'hidden_dim' (H), 'nomask' (bool)

Outputs results.json next to solution.py.
"""

import sys
import os
import json
import time
import importlib.util
import traceback
from pathlib import Path

# Development benchmark: single shape
N, C, H, B = 256, 128, 128, 1
WARMUP_ITERS = 10
BENCH_ITERS = 50
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2


def make_data(N, C, H=128, B=1, device="cuda", seed=42):
    """Build the (inp, mask, weights, cfg) tuple used by custom_kernel."""
    import torch
    torch.manual_seed(seed)
    inp = torch.randn(B, N, N, C, device=device, dtype=torch.float32)
    mask = torch.triu(torch.ones(B, N, N, device=device, dtype=torch.float32))
    weights = {
        "norm.weight":       torch.ones(C, device=device, dtype=torch.float32),
        "norm.bias":         torch.zeros(C, device=device, dtype=torch.float32),
        "left_proj.weight":  torch.randn(H, C, device=device, dtype=torch.float32) * 0.02,
        "right_proj.weight": torch.randn(H, C, device=device, dtype=torch.float32) * 0.02,
        "left_gate.weight":  torch.randn(H, C, device=device, dtype=torch.float32) * 0.02,
        "right_gate.weight": torch.randn(H, C, device=device, dtype=torch.float32) * 0.02,
        "out_gate.weight":   torch.randn(H, C, device=device, dtype=torch.float32) * 0.02,
        "to_out_norm.weight": torch.ones(H, device=device, dtype=torch.float32),
        "to_out_norm.bias":   torch.zeros(H, device=device, dtype=torch.float32),
        "to_out.weight":     torch.randn(C, H, device=device, dtype=torch.float32) * 0.02,
    }
    cfg = {"dim": C, "hidden_dim": H, "nomask": False}
    return inp, mask, weights, cfg


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_kernel"):
        raise AttributeError(f"solution.py must define custom_kernel(data), not found in {solution_path}")
    return module.custom_kernel


def reference_custom_kernel(data):
    import torch
    import torch.nn.functional as F
    inp, mask, weights, cfg = data
    dim = cfg["dim"]
    hidden_dim = cfg["hidden_dim"]
    nomask = cfg.get("nomask", True)
    B, N, _, C = inp.shape

    Z_norm = F.layer_norm(inp, [C],
                          weight=weights["norm.weight"],
                          bias=weights["norm.bias"])

    left = (Z_norm @ weights["left_proj.weight"].T) * torch.sigmoid(Z_norm @ weights["left_gate.weight"].T)
    right = (Z_norm @ weights["right_proj.weight"].T) * torch.sigmoid(Z_norm @ weights["right_gate.weight"].T)
    out_gate = torch.sigmoid(Z_norm @ weights["out_gate.weight"].T)

    left_bhnn = left.permute(0, 3, 1, 2)    # (B, H, N, N)
    right_bhnn = right.permute(0, 3, 1, 2)  # (B, H, N, N)

    if not nomask and mask is not None:
        left_bhnn  = left_bhnn  * mask.unsqueeze(1)
        right_bhnn = right_bhnn * mask.unsqueeze(1)

    left_mat = left_bhnn.reshape(B * hidden_dim, N, N)
    right_mat = right_bhnn.reshape(B * hidden_dim, N, N).transpose(1, 2)
    hidden = torch.bmm(left_mat, right_mat).reshape(B, hidden_dim, N, N)

    hidden_flat = hidden.permute(0, 2, 3, 1).reshape(-1, hidden_dim)
    hidden_norm = F.layer_norm(hidden_flat, [hidden_dim],
                               weight=weights["to_out_norm.weight"],
                               bias=weights["to_out_norm.bias"])
    out = (hidden_norm * out_gate.reshape(-1, hidden_dim)) @ weights["to_out.weight"].T
    return out.reshape(B, N, N, C)


def benchmark_fn(fn, data, warmup: int = 10, iters: int = 50) -> float:
    """Returns median runtime in milliseconds."""
    import torch
    for _ in range(warmup):
        fn(data)
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(data)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0  # median in ms


def run_evaluation(custom_kernel_fn) -> dict:
    import torch
    import time as _time
    for _attempt in range(4):
        if torch.cuda.is_available():
            break
        if _attempt < 3:
            _time.sleep(10)
    if not torch.cuda.is_available():
        return {
            "error": "CUDA not available — a GPU is required for kernel benchmarking",
            "score": 0.0,
        }

    data = make_data(N, C, H, B, device="cuda")
    inp = data[0]
    print(f"Shape: inp={tuple(inp.shape)}  hidden_dim={H}  device={inp.device}", flush=True)

    # Reference run (establishes baseline time)
    print("Benchmarking reference implementation...", flush=True)
    ref_time_ms = benchmark_fn(reference_custom_kernel, data, WARMUP_ITERS, BENCH_ITERS)
    ref_out = reference_custom_kernel(data).float()
    print(f"Reference: {ref_time_ms:.3f} ms", flush=True)

    # Correctness check
    print("Running correctness check...", flush=True)
    try:
        sol_out = custom_kernel_fn(data).float()
    except Exception as e:
        return {"error": f"custom_kernel() raised an exception: {e}\n{traceback.format_exc()}", "score": 0.0}

    if sol_out.shape != ref_out.shape:
        return {
            "error": f"Shape mismatch: expected {tuple(ref_out.shape)}, got {tuple(sol_out.shape)}",
            "score": 0.0,
        }

    if not torch.isfinite(sol_out).all():
        return {"error": "Non-finite values in output (NaN or Inf)", "score": 0.0}

    if not torch.allclose(ref_out, sol_out, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
        max_diff = (ref_out - sol_out).abs().max().item()
        mean_diff = (ref_out - sol_out).abs().mean().item()
        return {
            "error": f"Correctness check failed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} "
                     f"(atol={CORRECTNESS_ATOL}, rtol={CORRECTNESS_RTOL})",
            "score": 0.0,
        }
    print("Correctness check passed.", flush=True)

    # Solution benchmark
    print("Benchmarking solution...", flush=True)
    sol_time_ms = benchmark_fn(custom_kernel_fn, data, WARMUP_ITERS, BENCH_ITERS)
    print(f"Solution:  {sol_time_ms:.3f} ms", flush=True)

    speedup = ref_time_ms / sol_time_ms

    return {
        "ref_time_ms": ref_time_ms,
        "sol_time_ms": sol_time_ms,
        "speedup": speedup,
        "score": speedup,
        "lower_is_better": False,
        "error": None,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py solution.py", file=sys.stderr)
        sys.exit(1)

    solution_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(solution_path):
        result = {"error": f"File not found: {solution_path}", "score": 0.0}
    else:
        print(f"Loading solution from: {solution_path}", flush=True)
        try:
            custom_kernel_fn = load_solution(solution_path)
        except Exception as e:
            result = {"error": f"Failed to load solution: {e}", "score": 0.0}
        else:
            try:
                result = run_evaluation(custom_kernel_fn)
            except Exception as e:
                result = {
                    "error": f"Evaluation failed: {e}\n{traceback.format_exc()}",
                    "score": 0.0,
                }

    result["accuracy"] = result.get("score", 0.0)
    result["lower_is_better"] = False

    display = {k: v for k, v in result.items() if k != "solution_code"}
    print("\n=== EVALUATION RESULT ===")
    print(json.dumps(display, indent=2))
    print("=========================")

    if result.get("score", 0.0) > 0:
        print(f"\nSPEEDUP: {result['speedup']:.2f}x")
        print(f"SCORE:   {result['score']:.4f}")
        print(f"Reference: {result.get('ref_time_ms', 0):.3f} ms")
        print(f"Solution:  {result.get('sol_time_ms', 0):.3f} ms")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")

    print(f"RESULT_JSON:{json.dumps({k: v for k, v in result.items() if k != 'solution_code'})}")
    sys.exit(0)


if __name__ == "__main__":
    main()
