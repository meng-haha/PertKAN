#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
SCRIPT="$ROOT_DIR/main_deepce.py"

# Real filenames from `data/train_deep.sh`
DRUG_FILE="$ROOT_DIR/data/drug_smiles.txt"
GENE_FILE="$ROOT_DIR/data/DeepCE_gene_vectors.csv"
TRAIN_FILE="$ROOT_DIR/data/train.csv"
DEV_FILE="$ROOT_DIR/data/dev.csv"
TEST_FILE="$ROOT_DIR/data/test.csv"
CELL_FILE="$ROOT_DIR/data/cell_features.csv"

COMMON_ARGS=(
  --drug_file "$DRUG_FILE"
  --gene_file "$GENE_FILE"
  --train_file "$TRAIN_FILE"
  --dev_file "$DEV_FILE"
  --test_file "$TEST_FILE"
  --cell_file "$CELL_FILE"
  --dropout 0.3
  --batch_size 1
  --max_epoch 50
  --save_best
  --save_last
  --deg_topk 50,100,200
)

usage() {
  cat <<'EOF'
Usage:
  bash run_abcd_experiments.sh [A|B|C|D]
  bash run_abcd_experiments.sh all

Examples:
  bash run_abcd_experiments.sh A
  bash run_abcd_experiments.sh B
  bash run_abcd_experiments.sh all
EOF
}

run_exp() {
  local name="$1"
  shift
  echo ""
  echo "===== Running Experiment ${name} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -u "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_a() {
  run_exp A \
    --save_dir deepce_out_A \
    --best_metric f1 \
    --grad_accum_steps 4 \
    --grad_clip 1.0 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --loss_mode weighted_ce \
    --loss_down_weight 2.0 \
    --loss_none_weight 1.0 \
    --loss_up_weight 2.0 \
    --loss_focal_gamma 2.5 \
    --loss_down_up_miss_scale 1.0 \
    --train_sampler weighted \
    --sampler_power 1.0 \
    --seed 343
}

run_b() {
  run_exp B \
    --save_dir deepce_out_B \
    --best_metric f1 \
    --grad_accum_steps 8 \
    --grad_clip 1.0 \
    --lr 3e-4 \
    --weight_decay 5e-3 \
    --loss_mode focal+penalty \
    --loss_down_weight 3.0 \
    --loss_none_weight 0.9 \
    --loss_up_weight 3.0 \
    --loss_focal_gamma 3.0 \
    --loss_down_up_miss_scale 1.4 \
    --train_sampler weighted \
    --sampler_power 1.5 \
    --seed 343
}

run_c() {
  run_exp C \
    --save_dir deepce_out_C \
    --best_metric f1 \
    --grad_accum_steps 4 \
    --grad_clip 1.0 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --loss_mode weighted_ce \
    --loss_down_weight 2.0 \
    --loss_none_weight 1.0 \
    --loss_up_weight 2.0 \
    --loss_focal_gamma 1.5 \
    --loss_down_up_miss_scale 1.0 \
    --train_sampler weighted \
    --sampler_power 2.0 \
    --seed 343 \
    --cell_ids A375,MCF7,HT29
}

run_d() {
  run_exp D \
    --save_dir deepce_out_D \
    --best_metric f1 \
    --grad_accum_steps 4 \
    --grad_clip 1.0 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --loss_mode weighted_ce \
    --loss_down_weight 2.0 \
    --loss_none_weight 1.0 \
    --loss_up_weight 2.0 \
    --loss_focal_gamma 2.5 \
    --loss_down_up_miss_scale 1.0 \
    --train_sampler weighted \
    --sampler_power 1.5 \
    --seed 343
}

mode="${1:-all}"
case "$mode" in
  A|a)
    run_a
    ;;
  B|b)
    run_b
    ;;
  C|c)
    run_c
    ;;
  D|d)
    run_d
    ;;
  all|ALL)
    run_a
    run_b
    run_c
    run_d
    echo ""
    echo "All experiments completed."
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode"
    usage
    exit 1
    ;;
esac
