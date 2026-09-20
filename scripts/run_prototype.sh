#!/usr/bin/env bash
set -euo pipefail
# Activate ReConGS before invoking this script.
methods=("$@")
if (( $# == 0 )); then methods=(gradhet); fi
for method in "${methods[@]}"; do
  if [[ "$method" == legacy ]]; then
    python main.py --pipeline=ReCon-GS --config=config/prototype.yaml experiment_name=prototype_legacy module.deformation.hierarchy_mode=legacy module.deformation.mask_build_step=100000
  else
    python main.py --pipeline=ReCon-GS --config=config/prototype.yaml "experiment_name=prototype_${method}_v2" "module.deformation.refinement_mode=$method"
  fi
done
python scripts/summarize_prototype.py
