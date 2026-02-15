# Pre-Submission Checklist

## Code integrity

- [ ] `python -m py_compile models/inp.py`
- [ ] `python -m py_compile models/loss.py`
- [ ] `python -m py_compile models/train.py`
- [ ] `python -m py_compile models/modules.py`
- [ ] `python -m py_compile models/knowledge_corruption.py`
- [ ] `python -m py_compile config.py`
- [ ] `python -m py_compile dataset/dataset.py`
- [ ] `python -m py_compile dataset/utils.py`
- [ ] `python models/knowledge_corruption.py` prints `ALL PASS`

## Notebooks run end-to-end

- [ ] `notebooks/01_stepA_train.ipynb` completes without errors
- [ ] `notebooks/02_stepB_eval.ipynb` completes without errors
- [ ] `notebooks/03_figures.ipynb` completes without errors

## Output artifacts present

- [ ] 9 run directories in `outputs/` (3 regimes x 3 seeds)
- [ ] Each run dir contains: `model_best.pt`, `config.toml`, `metrics.jsonl`
- [ ] `outputs/stepB_eval.tsv` exists and has 36 rows (9 runs x 4 corruption regimes)
- [ ] `outputs/stepB_eval.json` exists

## Figures generated

- [ ] `outputs/paper_figures/main/fig1_mean_nll_grid.pdf`
- [ ] `outputs/paper_figures/main/fig2_context_curves.pdf`
- [ ] `outputs/paper_figures/main/fig3_delta_vs_baseline.pdf`
- [ ] `outputs/paper_figures/appendix/figA1_training_nll.pdf`
- [ ] `outputs/paper_figures/appendix/figA1_training_nll_zoom.pdf`
- [ ] `outputs/paper_figures/appendix/figA2_alignment_diagnostics.pdf`

## Documentation

- [ ] `README.md` updated with INP-Align overview
- [ ] `RUNBOOK.md` present with reproduction instructions
- [ ] No mention of temperature dataset as MVP target
- [ ] Metric terminology consistent: "Predictive NLL (IS-NLL)" throughout

## Repository hygiene

- [ ] `git status` is clean (all changes committed)
- [ ] `outputs/.gitignore` prevents checkpoints from being tracked
- [ ] No secrets, credentials, or large binaries committed

## Optional: quick CI-like sanity check

```bash
# Compile all source files
for f in models/inp.py models/loss.py models/train.py models/modules.py \
         models/knowledge_corruption.py config.py dataset/dataset.py \
         dataset/utils.py; do
    python -m py_compile "$f" && echo "OK  $f" || echo "FAIL $f"
done

# Run corruption self-test
python models/knowledge_corruption.py
```
