#!/bin/bash
#SBATCH --job-name=difflogic_run
#SBATCH --output=logs/difflogic_%j.log  # Combined output and error
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu08,tikgpu10
#SBATCH --time=2-00:00:00

# Debugging setup
export CUDA_VISIBLE_DEVICES=0

# Configure to suppress font warnings but keep wandb logging
export MPLBACKEND=agg  # Use non-interactive backend
export PYTHONWARNINGS="ignore::UserWarning"  # Only suppress UserWarnings

echo "=== HOST INFORMATION ==="
hostname
nvidia-smi
echo "======================="

# Run the job for factor sweep from 0.00 to 1.00 (step 0.01)

  singularity exec --nv \
    --bind $(pwd):/mnt \
    --bind $(pwd)/data:/data \
    Singularity/pytorch2.4.0-cuda12.4-universal.sif \
    python /mnt/src/main.py --config /mnt/configs/wmt/unsynced_recurrent_difflogic.json
