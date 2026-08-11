#!/bin/bash

set -e
set -x

variant=$1
dataset=$2
memax_weight=$3
exp_prefix=$4

mkdir -p /userhome/cs/mkyahx/dev_outputs
source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /userhome/cs/mkyahx/SimGCD/

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 1 2; do
    PYTHONHASHSEED=$seed CUDA_VISIBLE_DEVICES=0 python "train_asymmetric_${variant}_repro.py" \
        --dataset_name "$dataset" \
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
        --memax_weight "$memax_weight" \
        --exp_name "${exp_prefix}_seed_${seed}" \
        --seed "$seed" \
        --mask_root /userhome/cs/mkyahx/SimGCD/masks \
        --max_foreground_tokens 128
done

conda deactivate
echo "Finish"
