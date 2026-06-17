#!/bin/bash
#SBATCH --job-name=simgcd
#SBATCH --partition=gpu
#SBATCH --account=stat_han
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=~/logs/simgcd_cub_v2_2d_ada_%j.out
#SBATCH --error=~/logs/simgcd_cub_v2_2d_ada_%j.err


set -e
set -x

source /home/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /home/mkyahx/SimGCD/
TRAIN_SEED=0
export PYTHONHASHSEED="${TRAIN_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# MASK_ARGS=()
MASK_ARGS=(
    --use_mask
    --mask_root '/home/mkyahx/masks_cub/gt_bbox'
    --mask_image_root '/home/mkyahx/data/CUB_200_2011/images'
    --mask_threshold 0.5
    --missing_mask 'error'
)

python /home/mkyahx/SimGCD/train_last_dbb_2_adaptive.py \
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
    --seed "${TRAIN_SEED}" \
    --deterministic \
    --exp_name cub_simgcd_dinov2_2d_mask \
    "${MASK_ARGS[@]}"

conda deactivate
echo "Finish"

    # --backbone_repo 'facebookresearch/dino:main' \
    # --backbone_name 'dino_vitb16' \
    # --backbone_weights '/home/mkyahx/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth' \
