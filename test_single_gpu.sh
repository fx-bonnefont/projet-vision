#!/bin/bash
# Quick test script for single-GPU training.
# This script runs a single epoch with a small model to verify the setup.

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================="
echo "Running Single-GPU Sanity Check"
echo "================================="

# Parameters for a quick test. Can be overridden by environment variables.
MODEL_SIZE=${MODEL_SIZE:-"vit-small"}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-1}
DATA_DIR=${DATA_DIR:-"segdata/DOTA/DOTA_PLANES_TILED"}

echo "Model: $MODEL_SIZE"
echo "Epochs: $EPOCHS"
echo "Data Dir: $DATA_DIR"
echo "---------------------------------"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

# Run training with full output visible
python train.py \
    --model_size "$MODEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --data_dir "$DATA_DIR" \
    --num_workers 2

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Test finished successfully!"
  echo "Duration: ${MINUTES}m ${SECONDS}s"
  echo "================================="
else
  echo "Test failed with exit code: $EXIT_CODE"
  echo "Duration: ${MINUTES}m ${SECONDS}s"
  echo "================================="
fi

exit $EXIT_CODE
