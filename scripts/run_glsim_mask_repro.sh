#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <seed> --mask_root <path> [train_glsim_mask_repro.py arguments]" >&2
  exit 2
fi

seed="$1"
shift

export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train_glsim_mask_repro.py --seed "$seed" "$@"
