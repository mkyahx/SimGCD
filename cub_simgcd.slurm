#!/bin/bash
#SBATCH --job-name=cub_simgcd
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
#SBATCH --output=~/logs/cub_simgcd_%j.out
#SBATCH --error=~/logs/cub_simgcd_%j.err


set -e
set -x

source /home/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /home/mkyahx/SimGCD/

TRAIN_SEED=0
export PYTHONHASHSEED="${TRAIN_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# CUDA_VISIBLE_DEVICES is set by SLURM via --gres=gpu:1, no need to override.
python /home/mkyahx/SimGCD/train.py \
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
    --threshold 0.2 \
    --seed "${TRAIN_SEED}" \
    --exp_name cub_simgcd

conda deactivate
echo "Finish"
