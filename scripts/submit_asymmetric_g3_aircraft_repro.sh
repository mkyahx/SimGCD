#!/bin/bash
#SBATCH --job-name=asymmetric_g3_aircraft_repro
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=/userhome/cs/mkyahx/dev_outputs/asymmetric_g3_aircraft_repro_%j.out
#SBATCH --error=/userhome/cs/mkyahx/dev_outputs/asymmetric_g3_aircraft_repro_%j.err

+set -e
set -x

mkdir -p /userhome/cs/mkyahx/dev_outputs
source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate simgcd
cd /userhome/cs/mkyahx/SimGCD/

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 1 2; do
    PYTHONHASHSEED=$seed CUDA_VISIBLE_DEVICES=0 python train_asymmetric_g3_repro.py \
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
        --exp_name aircraft_asymmetric_g3_seed_${seed} \
        --seed $seed \
        --mask_root /userhome/cs/mkyahx/SimGCD/masks \
        --max_foreground_tokens 128
done

conda deactivate
echo "Finish"
