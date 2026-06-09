#!/usr/bin/env python3
"""
Private evaluator for the gpu-kernel task.

Evaluates solutions over multiple shapes to measure generalization across
matrix sizes. The development benchmark uses a single shape (B=1, N=256, C=128, H=128);
this private evaluator tests N ∈ {128, 192, 256, 320, 384} with C=H=128, B=1.

Usage:
    python evaluate.py --gen-dir runs/run_1/gen_3
    python evaluate.py --run-dir runs/run_1
    python evaluate.py path/to/solution.py
"""

import sys
import os
import json
import time
import glob
import argparse
import importlib.util
import traceback
from pathlib import Path

C, H, B = 128, 128, 1
BENCHMARK_NS = [128, 192, 256, 320, 384]
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
        raise AttributeError("solution.py must define custom_kernel(data)")
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
    return times[len(times) // 2] * 1000.0


def evaluate_on_shape(custom_kernel_fn, N) -> dict:
    import torch
    label = f"N={N},C={C},H={H}"
    data = make_data(N, C, H, B, device="cuda")

    ref_out = reference_custom_kernel(data).float()
    try:
        sol_out = custom_kernel_fn(data).float()
    except Exception as e:
        return {"error": f"[{label}] custom_kernel() raised: {e}", "speedup": 0.0}

    if sol_out.shape != ref_out.shape:
        return {"error": f"[{label}] Shape mismatch: expected {tuple(ref_out.shape)}, got {tuple(sol_out.shape)}", "speedup": 0.0}
    if not torch.isfinite(sol_out).all():
        return {"error": f"[{label}] Non-finite values in output", "speedup": 0.0}
    if not torch.allclose(ref_out, sol_out, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
        max_diff = (ref_out - sol_out).abs().max().item()
        return {"error": f"[{label}] Correctness failed: max_diff={max_diff:.6f}", "speedup": 0.0}

    ref_ms = benchmark_fn(reference_custom_kernel, data, WARMUP_ITERS, BENCH_ITERS)
    sol_ms = benchmark_fn(custom_kernel_fn, data, WARMUP_ITERS, BENCH_ITERS)
    speedup = ref_ms / sol_ms

    print(f"  [{label}] ref={ref_ms:.3f}ms  sol={sol_ms:.3f}ms  speedup={speedup:.2f}x", flush=True)
    return {"ref_ms": ref_ms, "sol_ms": sol_ms, "speedup": speedup, "error": None}


def score_solution(solution_path: str) -> dict:
    import torch
    solution_path = os.path.abspath(solution_path)
    if not os.path.exists(solution_path):
        return {"error": f"File not found: {solution_path}", "score": 0.0}

    for _attempt in range(4):
        if torch.cuda.is_available():
            break
        if _attempt < 3:
            import time as _time
            _time.sleep(10)
    if not torch.cuda.is_available():
        return {"error": "CUDA not available", "score": 0.0}

    try:
        custom_kernel_fn = load_solution(solution_path)
    except Exception as e:
        return {"error": f"Failed to load solution: {e}", "score": 0.0}

    per_shape = {}
    for n in BENCHMARK_NS:
        label = f"{n}x{n}x{C}"
        per_shape[label] = evaluate_on_shape(custom_kernel_fn, n)

    valid_speedups = [r["speedup"] for r in per_shape.values() if r.get("error") is None and r.get("speedup", 0) > 0]

    if not valid_speedups:
        avg_speedup = 0.0
        error = "All shapes failed"
    else:
        import math
        log_sum = sum(math.log(s) for s in valid_speedups)
        avg_speedup = math.exp(log_sum / len(valid_speedups))
        error = None if len(valid_speedups) == len(BENCHMARK_NS) else f"Only {len(valid_speedups)}/{len(BENCHMARK_NS)} shapes succeeded"

    try:
        with open(solution_path) as f:
            solution_code = f.read()
    except Exception:
        solution_code = None

    return {
        "score": avg_speedup,
        "accuracy": avg_speedup,
        "lower_is_better": False,
        "per_shape": per_shape,
        "num_shapes": len(valid_speedups),
        "error": error,
        "solution_code": solution_code,
    }


def main():
    parser = argparse.ArgumentParser(description="Private evaluator — gpu-kernel task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Run directory (evaluates all gen_X/)")
    group.add_argument("--gen-dir", help="Single generation directory")
    group.add_argument("solution", nargs="?", help="Path to a single solution.py")
    args = parser.parse_args()

    if args.gen_dir:
        gen_dir = os.path.abspath(args.gen_dir)
        result = score_solution(os.path.join(gen_dir, "solution.py"))
        out_path = os.path.join(gen_dir, "private_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        display = {k: v for k, v in result.items() if k != "solution_code"}
        print(json.dumps(display, indent=2))
        print(f"\n[private] avg speedup: {result['score']:.4f}x  (written to {out_path})")

    elif args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        gen_dirs = sorted(glob.glob(os.path.join(run_dir, "gen_*")))
        if not gen_dirs:
            print(f"No gen_* directories found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        all_scores = {}
        for gen_dir in gen_dirs:
            gen_name = os.path.basename(gen_dir)
            print(f"\n[{gen_name}] Evaluating...")
            result = score_solution(os.path.join(gen_dir, "solution.py"))
            all_scores[gen_name] = result
            out_path = os.path.join(gen_dir, "private_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        output_path = os.path.join(run_dir, "private_scores.json")
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        print(f"\n=== PRIVATE SCORES SUMMARY ===")
        for gen, r in all_scores.items():
            if not r.get("error"):
                print(f"  {gen}: speedup={r['score']:.4f}x")
            else:
                print(f"  {gen}: FAILED — {r['error']}")
        print(f"\nResults saved to: {output_path}")

    else:
        result = score_solution(args.solution)
        display = {k: v for k, v in result.items() if k != "solution_code"}
        print(json.dumps(display, indent=2))
        print(f"\nGeometric mean speedup: {result['score']:.4f}x")
        with open("private_result.json", "w") as f:
            json.dump(display, f, indent=2)


if __name__ == "__main__":
    main()
