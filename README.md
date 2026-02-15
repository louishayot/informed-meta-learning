# Informed Meta-Learning with INPs

This repository contains the code to reproduce the results of the experiments presented in the paper:

[Towards Automated Knowledge Integration From Human-Interpretable Representations](https://openreview.net/forum?id=NTHMw8S1Ow) published at ICLR 2025

For citations, use the following:
```
@inproceedings{
kobalczyk2025towards,
title={Towards Automated Knowledge Integration From Human-Interpretable Representations},
author={Katarzyna Kobalczyk and Mihaela van der Schaar},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=NTHMw8S1Ow}
}
```

---

## INP-Align Extension (MVP)

This branch adds **INP-Align**, a contrastive alignment extension to the
Informed Neural Process (INP) framework. It introduces a symmetric InfoNCE
loss that encourages alignment between the knowledge embedding and the
latent task representation, and evaluates robustness under structured
knowledge corruption.

The MVP is fully notebook-driven, uses the **set-trending-sinusoids** dataset
with **abc2** knowledge (numeric parameter triplets), and reports
**Predictive NLL (IS-NLL)** as the primary metric, kept strictly separate
from any alignment penalty.

See [RUNBOOK.md](RUNBOOK.md) for detailed reproduction instructions.

---

## Setup

```bash
conda env create -f environment.yaml
conda activate inps
```

**Optional dependencies:** `transformers` and `datasets` are only required for
`text_encoder="roberta"` and `knowledge_type="llama_embed"` respectively. The
MVP uses neither; both are lazily imported.

---

## Reproducibility (Notebooks)

The full pipeline is executed via three notebooks, run in order:

1. **[notebooks/01_stepA_train.ipynb](notebooks/01_stepA_train.ipynb)** --
   Train 3 regimes x 3 seeds on set-trending-sinusoids with abc2 knowledge.
2. **[notebooks/02_stepB_eval.ipynb](notebooks/02_stepB_eval.ipynb)** --
   Evaluate all checkpoints under knowledge corruption (clean / noisy / permuted).
3. **[notebooks/03_figures.ipynb](notebooks/03_figures.ipynb)** --
   Generate paper-ready PDF figures from evaluation results.

Set `FAST_DEV = True` in notebook 01 for a quick smoke test (1 seed, 50 epochs).

---

## Outputs

After running all notebooks:

```
outputs/
    {run_name}/                     # one per trained model
        model_best.pt               # best checkpoint
        config.toml                 # training configuration
        metrics.jsonl               # per-iteration training metrics
        stepB_{mode}.json           # per-regime evaluation results
    stepB_eval.tsv                  # aggregated evaluation (all runs x regimes)
    stepB_eval.json                 # same in JSON format
    paper_figures/
        main/
            fig1_mean_nll_grid.pdf
            fig2_context_curves.pdf
            fig3_delta_vs_baseline.pdf
        appendix/
            figA1_training_nll.pdf
            figA1_training_nll_zoom.pdf
            figA2_alignment_diagnostics.pdf
```

---

## Method Variants

| Method | `alignment_mode` | `alignment_lambda` | `alignment_temperature` |
|--------|-----------------|-------------------|------------------------|
| INP (baseline) | `none` | 0.0 | -- |
| INP-Align (align to r_T) | `rT` | 0.1 | 0.1 |
| INP-Align (align to r_C) | `rC` | 0.01 | 0.2 |

---

## Upstream Experiments

The original experiments from the ICLR 2025 paper remain available:

- **Synthetic data:** `jobs/run_sinusoids.sh` + `evaluation/evaluate_sinusoids.ipynb`
- **Distribution shift:** `evaluation/evaluate_sinusoids_dist_shift.ipynb`
- **Temperatures:** `jobs/run_temperatures.sh` + `evaluation/evaluate_temperature.ipynb`
