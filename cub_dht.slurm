#!/bin/bash
#SBATCH --job-name=simgcd_dht
#SBATCH --partition=gpu
#SBATCH --account=stat_han
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=~/logs/simgcd_cub_dht_%j.out
#SBATCH --error=~/logs/simgcd_cub_dht_%j.err

set -e
set -x

source /home/mkyahx/miniconda3/etc/profile.d/conda.sh

conda activate simgcd

# ====================== CUDA 设置 ======================
echo "=== CUDA Setup ==="
module load cuda/12.4 2>/dev/null || module load cuda

export CUDA_VISIBLE_DEVICES=0

# ====================== 最终检查 ======================
echo "=== Final Check on Compute Node ==="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

# ====================== 训练命令 ======================
REPO_ROOT=/home/mkyahx/SimGCD
DHT_ROOT=/home/mkyahx/dHT
cd ${REPO_ROOT}

python train_dht.py \
    --dht_root ${DHT_ROOT} \
    --backbone_repo 'facebookresearch/dino:main' \
    --backbone_name 'dino_vitb16' \
    --dataset_name 'cub' \
    --batch_size 16 \
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
    --dht_num_tokens 196 \
    --dht_patch_size 16 \
    --dht_mask_mode none \
    --dht_fg_ratio 0.75 \
    --dht_fg_density 3.0 \
    --dht_min_fg_tokens 16 \
    --dht_min_bg_tokens 16 \
    --seed 0 \
    --deterministic \
    --exp_name cub_simgcd_dino_v1_dht_none