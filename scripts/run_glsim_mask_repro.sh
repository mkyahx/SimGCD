#!/bin/bash

set -e
set -x

# Replace this dummy path once with the shared TokenCut mask root on the cluster.
SEED=1
MASK_ROOT="/path/to/tokencut/masks"

if [[ "$MASK_ROOT" == "/path/to/"* ]]; then
  echo "Set MASK_ROOT in $0 before running." >&2
  exit 2
fi

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES=0 python train_glsim_mask_repro.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --exp_name cub_glsim_mask_repro \
    --seed "$SEED" \
    --mask_root "$MASK_ROOT"
