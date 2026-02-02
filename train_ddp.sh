#!/bin/bash
#SBATCH --job-name=segdino_train      # Job name
#SBATCH --output=logs/slurm_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_%j.err     # Standard error log
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=2           # 2 GPUs = 2 tasks
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --cpus-per-task=8             # CPUs per task (for data loading)
#SBATCH --mem=64G                     # Total memory
#SBATCH --time=24:00:00               # Max runtime (24 hours)
#SBATCH --partition=3090              # GPU partition for NVIDIA 3090 GPUs

# ============================================
# SLURM Job Configuration for SegDino DDP Training
# ============================================
#
# This script launches distributed training on 2x NVIDIA 3090 GPUs
# using PyTorch DistributedDataParallel (DDP).
#
# Usage:
#   sbatch train_ddp.sh
#
# Or with custom parameters:
#   sbatch --export=ALL,MODEL_SIZE=vit-large,BATCH_SIZE=16,EPOCHS=25 train_ddp.sh
#
# ============================================

# Exit on any error
set -e

# Print job info
echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "============================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# Environment Setup
# ============================================

# 1. Configure CUDA Environment
# This cluster uses `export` instead of `module load`.
echo "Configuring CUDA 11.6 environment..."
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

# 2. Activate or create Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating with uv..."
    # Ensure python3.12 is available or adjust python version
    uv venv python3.12 .venv
    echo "Virtual environment created. Installing dependencies with uv..."
    source .venv/bin/activate
    uv sync --quiet
    echo "Dependencies installed."
else
    echo "Activating existing virtual environment..."
    source .venv/bin/activate
fi

# Verify CUDA is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Num GPUs: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch or CUDA not properly configured"
    exit 1
fi

# ============================================
# Training Parameters
# ============================================

# Default parameters (can be overridden via --export)
MODEL_SIZE=${MODEL_SIZE:-"vit-large"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-25}
LR=${LR:-5e-4}
NUM_WORKERS=${NUM_WORKERS:-8}
DATA_DIR=${DATA_DIR:-"segdata/DOTA/DOTA_PLANES_TILED"}

echo ""
echo "Training Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Batch per GPU: $BATCH_SIZE"
echo "  Effective Batch: $((BATCH_SIZE * 2))"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Num Workers: $NUM_WORKERS"
echo "  Data Dir: $DATA_DIR"
echo "============================================"
echo ""

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please set DATA_DIR environment variable or edit this script"
    exit 1
fi

# ============================================
# DDP Environment Variables (SLURM)
# ============================================

# SLURM automatically sets:
# - SLURM_PROCID (rank)
# - SLURM_LOCALID (local rank)
# - SLURM_NTASKS (world size)
# - SLURM_NODELIST

# Setup master address for communication
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "DDP Configuration:"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $SLURM_NTASKS"
echo "  Num Nodes: $SLURM_JOB_NUM_NODES"
echo "============================================"
echo ""

# ============================================
# Launch Training with srun
# ============================================

# srun will launch one process per task (2 tasks = 2 GPUs)
# Each process will automatically get SLURM_PROCID and SLURM_LOCALID

srun python train.py \
    --data_dir "$DATA_DIR" \
    --model_size "$MODEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "============================================"

exit $EXIT_CODE
