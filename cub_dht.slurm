#!/bin/bash
#SBATCH --job-name=simgcd_dht
#SBATCH --partition=gpu
#SBATCH --account=stat_han
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=~/logs/simgcd_cub_dht_%j.out
#SBATCH --error=~/logs/simgcd_cub_dht_%j.err

set -e
set -x

source /home/mkyahx/miniconda3/etc/profile.d/conda.sh

# dHT requires PyTorch >= 2.7. Override when your environment has another name:
#   DHT_CONDA_ENV=my_dht_env sbatch cub_dht.slurm
DHT_CONDA_ENV="${DHT_CONDA_ENV:-simgcd}"
conda activate "${DHT_CONDA_ENV}"

REPO_ROOT="${REPO_ROOT:-/home/mkyahx/SimGCD}"
DHT_ROOT="${DHT_ROOT:-/home/mkyahx/dHT}"
MASK_ROOT="${MASK_ROOT:-/home/mkyahx/masks_cub/CUB_200_2011_tokencut_masks_vit_base}"
MASK_IMAGE_ROOT="${MASK_IMAGE_ROOT:-/lustre1/g/stat_han/datasets/cub/CUB_200_2011/images}"

cd "${REPO_ROOT}"

TRAIN_SEED="${TRAIN_SEED:-0}"
export PYTHONHASHSEED="${TRAIN_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}:${DHT_ROOT}:${PYTHONPATH:-}"

# Modes:
#   none          : dHT only; TokenCut masks are not read.
#   fixed         : fixed foreground ratio controlled by DHT_FG_RATIO.
#   area_weighted : allocation follows mask area and DHT_FG_DENSITY.
DHT_MASK_MODE="${DHT_MASK_MODE:-none}"
DHT_FG_RATIO="${DHT_FG_RATIO:-0.75}"
DHT_FG_DENSITY="${DHT_FG_DENSITY:-3.0}"

MASK_ARGS=()
if [[ "${DHT_MASK_MODE}" != "none" ]]; then
    MASK_ARGS=(
        --mask_root "${MASK_ROOT}"
        --mask_image_root "${MASK_IMAGE_ROOT}"
        --mask_threshold 0.5
        --missing_mask error
    )
fi

# dHT constructs pixel-level hierarchies and is substantially heavier than
# the patch-token baseline. Start with batch size 16 on a 24 GB GPU and raise
# it only after checking peak memory.
python "${REPO_ROOT}/train_dht.py" \
    --dht_root "${DHT_ROOT}" \
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
    --dht_mask_mode "${DHT_MASK_MODE}" \
    --dht_fg_ratio "${DHT_FG_RATIO}" \
    --dht_fg_density "${DHT_FG_DENSITY}" \
    --dht_min_fg_tokens 16 \
    --dht_min_bg_tokens 16 \
    --seed "${TRAIN_SEED}" \
    --deterministic \
    --exp_name "cub_simgcd_dino_v1_dht_${DHT_MASK_MODE}" \
    "${MASK_ARGS[@]}"

conda deactivate
echo "Finish"
