#!/usr/bin/env bash
#SBATCH --job-name=frame-attack
#SBATCH --partition=P100 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/frame_attack_%j.out
#SBATCH --error=slurm_logs/frame_attack_%j.err

# export variables (in bashrc
# export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
# export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Launch tmux
# mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
# reconnect to tmux session after ssh drop out
# tmux ls
# tmux attach -t <session_name_or_id>


# from local computer open ssh tunnel
#ssh -N -J gpu -L 5000:127.0.0.1:5001 gpu 


set -euo pipefail
mkdir -p cd

# Activate environment and move to project directory.
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
source "${VENV_ACTIVATE:-vision/bin/activate}"

# Prevent OpenBLAS / NumPy import failures on constrained HPC jobs.
# export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
# export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"


# Launch training. Pass-through args: sbatch run_frame_attack.sbatch -- --arg value
python frame_attack.py --experiment-name patch-attack-remote "$@"
