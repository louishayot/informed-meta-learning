# 1. Création du dossier pour les logs
RUNS_DIR="runs_logs"
mkdir -p "$RUNS_DIR"

# 2. Définition propre de la fonction
run_one() {
    local tag=$1
    shift
    echo "----------------------------------------"
    echo "=== STARTING: $tag ==="
    echo "----------------------------------------"
    # On appelle le script python avec les arguments restants ($@)
    python models/train.py --run_name_prefix "$tag" "$@" 2>&1 | tee -a "$RUNS_DIR/${tag}.log"
    echo "=== DONE: $tag ===" | tee -a "$RUNS_DIR/${tag}.log"
}

# 3. Lancement des expériences
# --- BASELINE (lambda_align = 0) ---
run_one "P1_baseline_seed0" --seed 0 --lambda_align 0.0
run_one "P1_baseline_seed1" --seed 1 --lambda_align 0.0
run_one "P1_baseline_seed2" --seed 2 --lambda_align 0.0

# --- ALIGN (lambda_align = 0.1) ---
run_one "P1_align_l0p1_tau0p1_rT_seed0" --seed 0 --lambda_align 0.1 --align_temperature 0.1 --align_dim 64 --align_use_rT true
run_one "P1_align_l0p1_tau0p1_rT_seed1" --seed 1 --lambda_align 0.1 --align_temperature 0.1 --align_dim 64 --align_use_rT true
run_one "P1_align_l0p1_tau0p1_rT_seed2" --seed 2 --lambda_align 0.1 --align_temperature 0.1 --align_dim 64 --align_use_rT true

