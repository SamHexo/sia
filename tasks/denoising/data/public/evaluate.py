#!/usr/bin/env python3
"""
Evaluate a custom_denoise solution against the pancreas benchmark.

Usage:
    python evaluate.py solution.py

The solution.py must define a top-level `custom_denoise(X, **kwargs)` function.

Outputs a JSON result to stdout and exits 0 on success, 1 on failure.
"""

import sys
import os
import json
import time
import argparse
import importlib.util
import traceback
from pathlib import Path

# Make the repo root importable so we can pull from tasks.denoising._vendor.*
# regardless of where this script is invoked from. parents[4] resolves to
# the repo root (.../sia/tasks/denoising/data/public/evaluate.py).
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    }
}


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_denoise"):
        raise AttributeError(f"solution.py must define custom_denoise(), not found in {solution_path}")
    return module.custom_denoise


def evaluate_mse(test_data, denoised):
    import anndata, scanpy as sc, sklearn.metrics
    test_adata = anndata.AnnData(X=test_data.copy())
    denoised_adata = anndata.AnnData(X=denoised.copy())
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)
    return float(sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X))


def evaluate_poisson(train_data, test_data, denoised):
    import numpy as np
    from molecular_cross_validation.mcv_sweep import poisson_nll_loss
    import scprep
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    denoised_scaled = denoised_X * test_X.sum() / max(train_data.sum(), 1e-12)
    return float(poisson_nll_loss(test_X, denoised_scaled).mean())


V_BASELINES = BASELINES


def _split_data(adata, seed: int = 0):
    import scipy.sparse, numpy as np
    import molecular_cross_validation.util
    random_state = np.random.RandomState(seed)
    X = np.array(adata.X.todense()) if scipy.sparse.issparse(adata.X) else adata.X
    if not np.allclose(X, X.astype(int)):
        raise TypeError("Molecular cross-validation requires integer count data.")
    X = X.astype(int)
    X_train, X_test = molecular_cross_validation.util.split_molecules(X, 0.9, 0.0, random_state)
    is_missing = X_train.sum(axis=0) == 0
    X_train, X_test = X_train[:, ~is_missing], X_test[:, ~is_missing]
    adata = adata[:, ~is_missing].copy()
    adata.obsm["train"] = scipy.sparse.csr_matrix(X_train).astype(float)
    adata.obsm["test"] = scipy.sparse.csr_matrix(X_test).astype(float)
    return adata


def run_evaluation(custom_denoise_fn, seed=42, iteration_id: int = 0):
    import numpy as np
    import anndata as ad
    import scprep

    data_path = Path(__file__).parent / "pancreas.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Pancreas dataset not found at {data_path}. "
            "Run: bash tasks/denoising/download_data.sh pancreas"
        )
    adata = ad.read_h5ad(data_path)
    adata = _split_data(adata, seed=seed)

    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])

    print(f"Data loaded: {X_train.shape[0]} cells x {X_train.shape[1]} genes", flush=True)
    print(f"Running custom_denoise...", flush=True)

    t0 = time.time()
    Y_denoised = custom_denoise_fn(X_train, random_state=seed)
    elapsed = time.time() - t0

    print(f"Finished in {elapsed:.1f}s", flush=True)

    Y_denoised = np.asarray(Y_denoised)

    if not np.isfinite(Y_denoised).all():
        return {"error": "Non-finite values in output", "score": 0.0, "mse": None, "poisson": None}
    if np.any(Y_denoised < 0):
        return {"error": "Negative values in output", "score": 0.0, "mse": None, "poisson": None}

    mse = evaluate_mse(X_test, Y_denoised)
    poisson = evaluate_poisson(X_train, X_test, Y_denoised)

    baseline = V_BASELINES["pancreas"]
    mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]

    mse_norm = (baseline["baseline_mse"] - mse) / mse_range if mse_range > 0 else 0
    mse_norm = max(0.0, min(1.0, mse_norm))

    poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range if poisson_range > 0 else 0
    poisson_norm = max(0.0, min(1.0, poisson_norm))

    score = (mse_norm + poisson_norm) / 2

    return {
        "score": score,
        "iteration_id": iteration_id,
        "lower_is_better": False,
        "mse": mse,
        "poisson": poisson,
        "mse_norm": mse_norm,
        "poisson_norm": poisson_norm,
        "elapsed_seconds": elapsed,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", help="Path to solution.py")
    parser.add_argument("--iteration_id", type=int, default=0,
                        help="Sequence number of this evaluation within the current generation")
    args = parser.parse_args()

    solution_path = os.path.abspath(args.solution)
    if not os.path.exists(solution_path):
        print(json.dumps({"error": f"File not found: {solution_path}", "score": 0.0}))
        sys.exit(1)

    print(f"Loading solution from: {solution_path}", flush=True)

    try:
        magic_fn = load_solution(solution_path)
    except Exception as e:
        result = {"error": f"Failed to load solution: {e}", "score": 0.0, "mse": None, "poisson": None}
        print(json.dumps(result))
        sys.exit(1)

    try:
        result = run_evaluation(magic_fn, iteration_id=args.iteration_id)
    except Exception as e:
        result = {
            "error": f"Evaluation failed: {e}\n{traceback.format_exc()}",
            "score": 0.0,
            "mse": None,
            "poisson": None,
        }

    result["accuracy"] = result.get("score", 0.0)

    # Print metrics to stdout (exclude solution_code — it's large)
    result_for_stdout = {k: v for k, v in result.items() if k != "solution_code"}
    print("\n=== EVALUATION RESULT ===")
    print(json.dumps(result_for_stdout, indent=2))
    print("=========================")

    if result.get("score", 0) > 0:
        print(f"\nSCORE: {result['score']:.4f}")
        print(f"MSE: {result['mse']:.6f} (norm: {result.get('mse_norm', 0):.4f})")
        print(f"Poisson: {result['poisson']:.6f} (norm: {result.get('poisson_norm', 0):.4f})")
        print(f"Poisson norm: {result.get('poisson_norm', 0):.4f}")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")

    print(f"RESULT_JSON:{json.dumps({k: v for k, v in result.items() if k != 'solution_code'})}")
    sys.exit(0)


if __name__ == "__main__":
    main()
