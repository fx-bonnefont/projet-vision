#!/bin/bash
#SBATCH --job-name=dota_segmentation
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables
DATA_DIR="~/data/DOTA"
BACKBONE="dinov3_vitb16"
EPOCHS=100
BATCH_SIZE=8
LR="1e-3"
IMG_SIZE=768
OUTPUT_DIR="~/projet-vision/checkpoints"

# Activate environment (uv/venv)
source ~/projet-vision/.venv/bin/activate

# Navigate to project directory
cd ~/projet-vision

# Execute training with cache and larger crop size
srun python train.py \
    --data $DATA_DIR \
    --backbone $BACKBONE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --img-size $IMG_SIZE \
    --cache \
    --output $OUTPUT_DIR/model_vitb_768.pth

# Print job completion time
echo "Job finished at: $(date)"
