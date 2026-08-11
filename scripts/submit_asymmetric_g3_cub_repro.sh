#!/bin/bash
#SBATCH --job-name=asymmetric_g3_cub_repro
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=/userhome/cs/mkyahx/dev_outputs/asymmetric_g3_cub_repro_%j.out
#SBATCH --error=/userhome/cs/mkyahx/dev_outputs/asymmetric_g3_cub_repro_%j.err

exec bash "$(dirname "$0")/run_asymmetric_variant_repro.sh" g3 cub 2 cub_asymmetric_g3
