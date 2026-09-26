#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="/home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300"
output_dir="$repo_dir/output/v2_300"

if [[ -e "$output_dir/train.log" ]]; then
  echo "Refusing to mix a new V2 run with existing results: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
cd "$repo_dir"

conda run --no-capture-output -n s2gs python train.py \
  --config configs/dynerf_flame_steak.yaml \
  -s "$data_dir" \
  -m "$output_dir" \
  --max_frames 300 \
  --shared_motion \
  --shared_motion_levels 4 \
  --shared_motion_stage1_steps 100 \
  --shared_motion_stage2_steps 200 \
  --v2_mode \
  --v2_birth_steps 100 \
  --v2_max_new_points 128 \
  --quiet 2>&1 | tee "$output_dir/train.log"
