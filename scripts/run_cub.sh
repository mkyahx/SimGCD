#!/bin/bash

set -e
set -x

LAST_VIT_ARGS=()
if [ "${USE_LAST_VIT:-1}" = "0" ]; then
    LAST_VIT_ARGS+=(--no_last_vit)
else
    LAST_VIT_ARGS+=(--use_last_vit)
fi
if [ -n "${LAST_VIT_TOPK:-}" ]; then
    LAST_VIT_ARGS+=(--last_vit_topk "${LAST_VIT_TOPK}")
fi
if [ -n "${LAST_VIT_SIGMA:-}" ]; then
    LAST_VIT_ARGS+=(--last_vit_sigma "${LAST_VIT_SIGMA}")
fi
if [ -n "${LAST_VIT_EPS:-}" ]; then
    LAST_VIT_ARGS+=(--last_vit_eps "${LAST_VIT_EPS}")
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
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
    --exp_name cub_simgcd \
    "${LAST_VIT_ARGS[@]}"
