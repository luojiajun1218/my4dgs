#!/usr/bin/env bash
set -euo pipefail

# Usage: run_adaptive_gate_v7c.sh [run_name [resume_snapshot [max_frames]]]
# Defaults to output/v7c_300 and explicitly enables the 0.1 sq_res floor.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="/home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300"
run_name="${1:-v7c_300}"
resume_snapshot="${2:-}"
max_frames="${3:-300}"

if [[ "$run_name" == */* || "$run_name" == "." || "$run_name" == ".." ]]; then
  echo "Run name must be a single output-directory name: $run_name" >&2
  exit 1
fi

output_dir="$repo_dir/output/$run_name"
if [[ -e "$output_dir" ]]; then
  echo "Refusing to mix a new run with existing results: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
cd "$repo_dir"

command=(
  conda run --no-capture-output -n s2gs python -u train.py
  --config configs/dynerf_flame_steak.yaml
  -s "$data_dir"
  -m "$output_dir"
  --max_frames "$max_frames"
  --adaptive_gate_partition_v7b
  --adaptive_gate_sqres_floor 0.1
)
if [[ -n "$resume_snapshot" ]]; then
  command+=(--resume_frame1_snapshot "$resume_snapshot")
fi

"${command[@]}" 2>&1 | tee "$output_dir/train.log"
