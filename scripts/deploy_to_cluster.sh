#!/bin/bash

# Configuration
PROJECT_NAME="projet-vision"
REMOTE_USER="bonnefont-25"
REMOTE_HOST="gpu-gw"
LOCAL_PROJECT_DIR="$HOME/TP-REPOS/$PROJECT_NAME"
ARCHIVE_NAME="${PROJECT_NAME}-clean.tar.gz"

echo "Syncing code to cluster ($REMOTE_USER@$REMOTE_HOST)..."

# Sync using rsync (faster and handles overwrites better than tar+scp)
rsync -az --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='checkpoints' \
    --exclude='logs' \
    --exclude='*.pth' \
    --exclude='.env' \
    "$LOCAL_PROJECT_DIR/" "$REMOTE_USER@$REMOTE_HOST:~/$PROJECT_NAME/"

if [ $? -eq 0 ]; then
    echo "Sync complete!"
    echo "Next steps on the cluster:"
    echo "  ssh $REMOTE_USER@$REMOTE_HOST"
    echo "  cd $PROJECT_NAME"
    echo "  source .venv/bin/activate && uv pip install ."
else
    echo "Sync failed."
fi
