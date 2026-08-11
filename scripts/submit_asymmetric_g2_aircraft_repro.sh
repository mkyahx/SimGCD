#!/bin/bash
#SBATCH --job-name=asymmetric_g2_aircraft_repro
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=/userhome/cs/mkyahx/dev_outputs/asymmetric_g2_aircraft_repro_%j.out
#SBATCH --error=/userhome/cs/mkyahx/dev_outputs/asymmetric_g2_aircraft_repro_%j.err

exec bash "$(dirname "$0")/run_asymmetric_variant_repro.sh" g2 aircraft 1 aircraft_asymmetric_g2
