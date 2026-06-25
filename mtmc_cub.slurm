#!/bin/bash
#SBATCH --job-name=mtmc
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
#SBATCH --output=~/logs/mtmc_cub_%j.out
#SBATCH --error=~/logs/mtmc_cub_%j.err


set -e
set -x

source /home/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /home/mkyahx/SimGCD/

TRAIN_SEED=0
export PYTHONHASHSEED="${TRAIN_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Paper-faithful MTMC on DINOv1 ViT-B/16 (arXiv 2505.14044).
# Set --mtmc_weight 0 to obtain the no-MTMC baseline for comparison.
#
# The paper-faithful loss is -sum sigma_r on raw DINOv1 [CLS] tokens
# (norm ~ 60), giving a raw value ~300 for a typical unlabelled
# mini-batch. With the default mtmc_weight=0.05 the contribution is
# ~15, on the same order as cls_loss (~5) and me_max (~10). If your
# accuracy drops, sweep {0.01, 0.05, 0.1}; if SVD keeps failing on
# the raw features, add --mtmc_engineered to apply L2 row norm + 1/sqrt(B_u).
python /home/mkyahx/SimGCD/train_mtmc.py \
    --backbone 'dinov1' \
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
    --memax_weight 2.0 \
    --mtmc_weight 0.05 \
    --seed "${TRAIN_SEED}" \
    --deterministic \
    --exp_name mtmc_cub_simgcd_dinov1_lam0.05

conda deactivate
echo "Finish"
