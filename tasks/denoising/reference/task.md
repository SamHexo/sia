# Denoising Task

## Overview

Implement a `custom_denoise(X, **kwargs)` function that denoises single-cell RNA-seq count data. **The goal is to beat MAGIC (~0.61) with a novel approach — not to tune MAGIC parameters.**

Your solution is developed on pancreatic islet cells (`pancreas.h5ad`), but the final score is computed privately on two held-out datasets you never see: **PBMC** (human blood immune cells) and **Tabula** (multi-tissue atlas).

## Problem

Single-cell RNA-seq (scRNA-seq) data is noisy due to technical dropout and low capture efficiency. Given noisy count data, predict the true expression levels.

The key challenge: scRNA-seq dropout noise follows a Poisson-like process regardless of tissue type — a cell that captured 5% of its true transcripts looks the same whether it's a T-cell or a pancreatic beta cell. Your denoiser should exploit this shared statistical structure, not pancreas-specific biology.

## Function Signature

```python
def custom_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # Input X: numpy array (n_cells, n_genes) — raw count data
    # Return: denoised counts, same shape, non-negative floats
    return denoised_X
```

## Scoring

Two metrics, each normalized to [0, 1] and averaged:

```
mse_norm     = (0.304721 - mse)     / (0.304721 - 0.000000)
poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)
score        = (mse_norm + poisson_norm) / 2
```

Higher is better (range 0–1).

Reference values (pancreas):
- `baseline_mse = 0.304721`, `baseline_poisson = 0.257575`  ← identity (no denoising)
- `perfect_mse = 0.000000`, `perfect_poisson = 0.031739`    ← oracle upper bound

MAGIC scores **~0.61** on pancreas and **~0.64** on private datasets.

## Available Libraries

numpy, scipy, sklearn, graphtools, scprep, scanpy, anndata, molecular_cross_validation, **magic-impute** (v3)

## Baseline (MAGIC — to beat, not to tune)

```python
import magic, numpy as np, scprep

def custom_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X_sqrt = np.sqrt(X)
    X_norm, libsize = scprep.normalize.library_size_normalize(X_sqrt, rescale=1, return_library_size=True)
    Y = magic.MAGIC(solver="approximate", verbose=False).fit_transform(X_norm, genes="all_genes")
    Y = np.asarray(Y) ** 2
    return np.maximum(Y * libsize[:, np.newaxis], 0)
```

Implement a **different** algorithm. The `magic-impute` library is available if useful for reference or hybridization, but your solution must not simply be MAGIC with different hyperparameters.

## Rules

1. Write your solution to the path you have been given — do not change it
2. Evaluate using `python {dataset_dir}/evaluate.py <your_candidate_path>`
3. No filesystem or network IO inside `custom_denoise`
4. No closures or lambdas — all helper functions must be top-level

## Dataset Directory Layout

```
{dataset_dir}/
├── pancreas.h5ad   ← development dataset (load with anndata.read_h5ad)
└── evaluate.py     ← evaluation script

{working_dir}/      ← your read/write workspace
```

To access the dataset directory use its absolute path explicitly (e.g. `ls {dataset_dir}`).

## Generalization

Methods that fit parameters *to the pancreas data* (autoencoders, tissue-specific models, hard-coded gene lists) will fail on PBMC and Tabula. Methods that use only the *structure of each dataset at inference time* (graph diffusion, PCA-based smoothing, low-rank approximation) generalize naturally.

Overfitting to pancreas-specific biology (e.g. hard-coded gene lists, tissue-specific priors) will hurt performance on held-out tissues.

## Domain Knowledge

- scRNA-seq dropout follows Poisson statistics — sqrt transform is variance-stabilizing
- Poisson loss is sensitive to small non-zero values — output must be non-negative
- Focus on MSE reduction while maintaining Poisson norm ≥ 0.97
- Why MAGIC works: graph diffusion over a k-NN graph propagates expression signal across similar cells, filling in dropouts. Any method with a similar inductive bias (manifold smoothing, low-rank approximation, graph convolution) can compete
- MAGIC pipeline: `sqrt(X)` → library-size normalize → diffuse over k-NN graph → square → rescale by libsize. Reversed normalization order (sqrt before library-size) is what achieves ~0.61
