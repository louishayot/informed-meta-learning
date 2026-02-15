# Runbook: INP-Align MVP Reproduction

This document provides step-by-step instructions to reproduce all results
from the INP-Align extension. The entire pipeline is notebook-driven.

---

## 1. Environment Setup

### Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate inps
```

### Minimal pip (if not using conda)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas matplotlib toml wandb optuna
```

**Python version:** 3.11 (tested), 3.10+ should work.

**Optional dependencies** (not required for the MVP):
- `transformers`: only needed if using `text_encoder="roberta"`.
- `datasets`: only needed if using `knowledge_type="llama_embed"`.

The MVP uses `text_encoder="set"` and `knowledge_type="abc2"`, so neither
is required. Both are lazily imported and will produce a clear error message
if needed but missing.

---

## 2. Canonical Reproduction Path

All commands assume you are in the repository root directory.

### Step A: Train models

Open and run all cells in:
```
notebooks/01_stepA_train.ipynb
```

This trains 9 models (3 regimes x 3 seeds) on the `set-trending-sinusoids`
dataset with `abc2` knowledge. Each run produces:
```
outputs/{run_name}/
    model_best.pt       # best checkpoint (by validation loss)
    config.toml         # full training configuration
    metrics.jsonl       # per-iteration metrics (train NLL, alignment, eval)
```

Run names:
```
baseline_seed0, baseline_seed1, baseline_seed2
aggressive_align_rT_seed0, aggressive_align_rT_seed1, aggressive_align_rT_seed2
safe_align_rC_seed0, safe_align_rC_seed1, safe_align_rC_seed2
```

### Step B: Evaluate under knowledge corruption

Open and run all cells in:
```
notebooks/02_stepB_eval.ipynb
```

This evaluates all 9 checkpoints under 4 corruption regimes at context sizes
|C| in {0, 3, 5, 10}. Outputs:
```
outputs/stepB_eval.tsv          # aggregated results (all runs x all regimes)
outputs/stepB_eval.json         # same data in JSON format
outputs/{run_name}/stepB_{mode}.json   # per-run per-regime results
```

### Step C: Generate paper-ready figures

Open and run all cells in:
```
notebooks/03_figures.ipynb
```

Outputs:
```
outputs/paper_figures/main/
    fig1_mean_nll_grid.pdf              # mean NLL per corruption (2x2 bar chart)
    fig2_context_curves.pdf             # NLL vs |C| per corruption (2x2 line plot)
    fig3_delta_vs_baseline.pdf          # robustness delta vs baseline

outputs/paper_figures/appendix/
    figA1_training_nll.pdf              # training NLL curves (full)
    figA1_training_nll_zoom.pdf         # training NLL curves (first 3000 iterations)
    figA2_alignment_diagnostics.pdf     # alignment loss, retrieval acc, cosine sim
```

---

## 3. Smoke Test (FAST_DEV)

For a quick end-to-end check without full training:

1. In `notebooks/01_stepA_train.ipynb`, set `FAST_DEV = True` in cell 2.
   This uses 1 seed and 50 epochs instead of 3 seeds and 1000 epochs.

2. Run Step A, then Step B (no changes needed), then Step C.

3. Verify PDFs are generated in `outputs/paper_figures/main/`.

Total time depends on hardware (CPU vs GPU, core count).

---

## 4. Full Reproduction

Use default notebook settings (no changes required):

| Parameter | Value |
|-----------|-------|
| Seeds | [0, 1, 2] |
| Epochs | 1000 |
| MAX_EVAL_BATCHES | 50 |
| EVAL_SEED | 42 |
| Context sizes | {0, 3, 5, 10} |

Total time depends on hardware. Full training is significantly longer than the smoke test.

---

## 5. Knowledge Corruption Protocol (abc2)

The `corrupt_knowledge()` function in `models/knowledge_corruption.py` implements
the strict abc2 corruption protocol. Knowledge tensors have shape `[B, 3, 4]`:
- 3 rows correspond to parameters (a, b, c).
- 4 columns: 3 one-hot indicator columns + 1 value column.
- Active rows have non-zero indicator; inactive rows are all zeros (masked by abc2 sampling).

### Corruption regimes

| Regime | `regime_str` | Description |
|--------|-------------|-------------|
| Clean | `"clean"` | No modification (returns a clone). |
| Noisy (low) | `"noisy_0.1"` | Gaussian noise on value column of active rows. |
| Noisy (high) | `"noisy_0.3"` | Gaussian noise on value column of active rows. |
| Permuted | `"permuted"` | Shuffle knowledge across batch dimension. |

### Noisy regime details

Noise is applied **only** to the value column (index 3) of **active** rows:

```
sigma_abs = sigma_rel * [2, 6, 2]
```

| sigma_rel | sigma_abs (a, b, c) |
|-----------|---------------------|
| 0.1 | (0.2, 0.6, 0.2) |
| 0.3 | (0.6, 1.8, 0.6) |

After adding noise, values are clamped to valid ranges:

| Parameter | Range |
|-----------|-------|
| a | [-1, 1] |
| b | [0, 6] |
| c | [-1, 1] |

Indicator columns are **never** modified. Inactive rows remain all-zeros.

### Evaluation protocol

- Corruption is applied **once per batch** and reused for all context sizes.
- Context indices are **pre-sampled once per batch** and reused across all
  corruption regimes (variance control).
- Permutation is deterministic via `torch.Generator().manual_seed(seed)`.

---

## 6. Metrics

### Predictive NLL (IS-NLL)

The primary evaluation metric is the **importance-sampled negative log-likelihood**:

```
IS-NLL = -log(1/S * sum_s w_s * p(y|z_s))
```

where `w_s = p(z_s|C) / q(z_s|C,T)` are importance weights and `S` is the
number of latent samples (`test_num_z_samples=32`).

**Why it can be negative:** IS-NLL is a log-likelihood estimate. For distributions
with high probability density (narrow, well-calibrated predictions), the
log-likelihood can be positive, making the negative log-likelihood negative.
This is normal and indicates good predictive performance.

### Training objective vs reported metric

| Quantity | Definition | Used for |
|----------|-----------|----------|
| Training loss | ELBO + lambda * alignment_loss | Gradient updates |
| Predictive NLL (IS-NLL) | NLL-only, no KL, no alignment penalty | All reported results |
| Alignment loss | Symmetric InfoNCE between k and r | Diagnostics only |

The alignment penalty is **never** included in reported NLL values.
In `metrics.jsonl`, `train_predictive_nll` is the NLL-only metric,
while `train_loss` includes both ELBO and alignment terms.

---

## 7. Method Variants

| Method | `alignment_mode` | `alignment_lambda` | `alignment_temperature` |
|--------|-----------------|-------------------|------------------------|
| INP (baseline) | `none` | 0.0 | -- |
| INP-Align (align to r_T) | `rT` | 0.1 | 0.1 |
| INP-Align (align to r_C) | `rC` | 0.01 | 0.2 |

- **r_T**: posterior mean q(z|C,T) -- alignment to target-informed representation.
- **r_C**: prior mean q(z|C) -- alignment to context-only representation.

---

## 8. Determinism and Reproducibility

### Training seeds

Each regime is trained with seeds 0, 1, 2. The seed controls:
- `torch.manual_seed(seed)`
- `np.random.seed(seed)`
- `random.seed(seed)`

### Evaluation seed

All evaluations use `EVAL_SEED = 42` for:
- Batch iteration order (dataloader)
- Context index sampling
- Corruption noise generation (`seed = EVAL_SEED + batch_idx`)

### Corruption determinism

- Noisy corruption uses `torch.Generator().manual_seed(seed)` per batch.
- Permuted corruption uses `torch.randperm(..., generator=generator)`.
- Same seed always produces the same corruption.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'transformers'`

This occurs if `models/modules.py` is imported and tries to load RoBERTa.
The MVP uses `text_encoder="set"`, which does not require `transformers`.
The import is lazy (inside `RoBERTa.__init__` only). If you see this error:
- Ensure you are using `text_encoder="set"` in your config.
- Or install: `pip install transformers~=4.44.2`

### `ModuleNotFoundError: No module named 'datasets'`

Same pattern. Only needed for `knowledge_type="llama_embed"`. The MVP uses
`knowledge_type="abc2"`. The import is lazy (inside the `llama_embed` branch
of `Temperatures.__init__`). Fix: `pip install datasets<4.0.0` if needed.

### Empty `figA2_alignment_diagnostics.pdf`

Alignment metrics (`train_alignment_loss`, `train_retrieval_acc`,
`train_mean_cosine`) are logged on **separate JSONL lines** from training NLL.
The notebook parses them separately. If figA2 is empty:
- Check that `metrics.jsonl` for alignment runs contains lines with
  `train_alignment_loss` (e.g., `grep train_alignment_loss outputs/aggressive_align_rT_seed0/metrics.jsonl`).
- Ensure you re-trained with alignment enabled (not a baseline-only run).

### Path issues

All notebooks detect the repository root by walking up from the current
directory until they find `config.py`. If this fails:
- Run notebooks from within the repository (any subdirectory works).
- Or manually set `REPO_ROOT` in cell 1.

### Kernel restart required after code changes

If you modify `.py` files (e.g., `models/knowledge_corruption.py`), restart
the Jupyter kernel before re-running notebooks. Python caches imported modules.

---

## 10. What to Submit

The minimum deliverable for review is the full repository contents:

```
├── config.py
├── dataset/
├── models/
├── notebooks/
│   ├── 01_stepA_train.ipynb
│   ├── 02_stepB_eval.ipynb
│   └── 03_figures.ipynb
├── outputs/
│   ├── {run_name}/          # 9 directories (3 regimes x 3 seeds)
│   ├── stepB_eval.tsv
│   ├── stepB_eval.json
│   └── paper_figures/       # main/ and appendix/ PDFs
├── environment.yaml
├── README.md
├── RUNBOOK.md
└── CHECKLIST.md
```

Before submitting, run through [CHECKLIST.md](CHECKLIST.md) to verify all
artifacts are present and code compiles without errors.
