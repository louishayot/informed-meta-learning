#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (so config.py writes the correct config.toml)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${REPO_ROOT}/runs_patch1_full/${STAMP}"
mkdir -p "$OUT_ROOT"

export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

# ---- base experiment settings (edit once here) ----
PROJECT="INPs_sinusoids"
DATASET="set-trending-sinusoids-dist-shift"
USE_KNOWLEDGE="true"
KNOW_TYPE="abc2"
TEXT_ENCODER="set"
NOISE="0.2"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
KDROP="${KDROP:-0.0}"          # Patch1: set 0.0 for stable align signal; later you can try 0.3
ALIGN_DIM="${ALIGN_DIM:-64}"

run_one () {
  local name="$1"
  local seed="$2"
  local lambda_align="$3"
  local tau="$4"
  local use_rT="$5"

  echo "----------------------------------------"
  echo "=== STARTING: ${name} ==="
  echo "----------------------------------------"

  local run_dir="$OUT_ROOT/$name"
  mkdir -p "$run_dir"/{wandb,logs,artifacts}

  # Store wandb for THIS run inside its own folder (super clean)
  export WANDB_DIR="$run_dir/wandb"
  export WANDB_NAME="$name"
  export WANDB_GROUP="patch1_full_${STAMP}"

  # Generate config.toml for this run (overwrites repo-root config.toml)
  python config.py \
    --project-name "$PROJECT" \
    --dataset "$DATASET" \
    --use-knowledge "$USE_KNOWLEDGE" \
    --knowledge-type "$KNOW_TYPE" \
    --text-encoder "$TEXT_ENCODER" \
    --noise "$NOISE" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --knowledge-dropout "$KDROP" \
    --seed "$seed" \
    --lambda-align "$lambda_align" \
    --align-temperature "$tau" \
    --align-dim "$ALIGN_DIM" \
    --align-use-rT "$use_rT" \
    --run-name-prefix "$name" \
    --run-name-suffix ""

  # Snapshot config + code version
  cp config.toml "$run_dir/config.toml"
  git rev-parse HEAD > "$run_dir/git_commit.txt"

  # Run training + save console log
  python models/train.py 2>&1 | tee "$run_dir/logs/console.log"

  echo "=== DONE: ${name} ==="
}

seeds=(0 1 2)

# Baseline INP (align OFF)
for s in "${seeds[@]}"; do
  run_one "P1_baseline_seed${s}" "$s" 0.0 0.1 true
done

# INP-Align (your first “paper” config)
for s in "${seeds[@]}"; do
  run_one "P1_align_l0p1_tau0p1_rT_seed${s}" "$s" 0.1 0.1 true
done
