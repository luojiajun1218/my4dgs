#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="/home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300"
max_frames="${1:-300}"
output_dir="$repo_dir/output/${2:-v6_300}"

if [[ -e "$output_dir/train.log" ]]; then
  echo "Refusing to mix a new run with existing results: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
cd "$repo_dir"

conda run --no-capture-output -n s2gs python -u train.py \
  --config configs/dynerf_flame_steak.yaml \
  -s "$data_dir" \
  -m "$output_dir" \
  --max_frames "$max_frames" \
  --adaptive_hierarchy \
  --hierarchy_levels 4 \
  --hierarchy_root_scale 2.0 \
  --hierarchy_stage1_steps 100 \
  --hierarchy_gs_steps 200 \
  --hierarchy_growth_steps 100 \
  --hierarchy_off_ratio 0.0 \
  --hierarchy_gate_warmup_steps 100 \
  --hierarchy_offset_gate_lr 0.005 \
  --hierarchy_offset_gate_lambda_l0 0.0005 \
  --hierarchy_offset_gate_lambda_l2 0.001 \
  2>&1 | tee "$output_dir/train.log"
