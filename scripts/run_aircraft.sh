#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'aircraft' \
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
    --memax_weight 1 \
    --exp_name aircraft_simgcd \
    --last_vit_mode fusion \
    --last_vit_alpha 0.1 \
    --last_vit_topk_ratio 0.5 \
    --last_vit_eps 1e-6
