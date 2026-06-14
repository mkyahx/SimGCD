#!/bin/bash
#SBATCH --job-name=simgcd
#SBATCH --partition=visualai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/simgcd_cub_v2_2d_ada_%j.out
#SBATCH --error=./logs/simgcd_cub_v2_2d_ada_%j.err


source ~/anaconda3/etc/profile.d/conda.sh
conda activate cpt

python train_last_dbb_2_adaptive.py \
    --backbone_repo 'facebookresearch/dinov2' \
    --backbone_name 'dinov2_vitb14' \
    --dataset_name 'cub' \
    --batch_size 64 \
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
    --sf_mask_mode 'adaptive' \
    --sf_adaptive_k 1.0 \
    --sf_topk 4 \
    --sf_vote_topk 4 \
    --exp_name cub_simgcd_dinov2_2d_ada

conda deactivate
echo "Finish"
