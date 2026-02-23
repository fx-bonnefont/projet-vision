#!/usr/bin/env bash
#SBATCH --job-name=frame-attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/frame_attack_%j.out
#SBATCH --error=slurm_logs/frame_attack_%j.err

# To create env
# python3 -m venv vision
# source vision/bin/activate
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# pip install -m requirements.txt
# £On local machine: 
# ssh -N -o ExitOnForwardFailure=yes -R 0.0.0.0:5500:127.0.0.1:5000 gpu

set -euo pipefail
mkdir -p slurm_logs


# 2) Activate environment and move to project directory.
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}" && source "${VENV_ACTIVATE:-vision/bin/activate}"

# 3) Launch training (override MLFLOW_TRACKING_URI if needed).
python frame_attack.py
