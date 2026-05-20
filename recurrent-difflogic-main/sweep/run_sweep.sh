#!/bin/bash
#SBATCH --job-name=diff_sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-256%16 
#SBATCH --output=sweep/logs/%A_%a.log
#SBATCH --exclude=tikgpu[08,10]

# Export environment variables
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
export MPLBACKEND=agg  # Use non-interactive backend
export PYTHONWARNINGS="ignore::UserWarning"  # Only suppress UserWarnings

# Print job info
echo "Running sweep agent - SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S")"
echo "User: simon-jonas-buehrer"
nvidia-smi -L

# Create necessary directories if they don't exist
mkdir -p sweep/tmp
mkdir -p sweep/logs
mkdir -p wandb_logs
mkdir -p wandb_config

# Run the sweep agent using Singularity
singularity exec --nv \
  --bind $(pwd):/mnt \
  --bind $(pwd)/data:/data \
  --bind $(pwd)/configs:/mnt/configs \
  --bind $(pwd)/wandb_logs:/mnt/wandb_logs \
  --bind $(pwd)/wandb_config:/mnt/wandb_config \
  Singularity/pytorch2.4.0-cuda12.4-universal.sif \
  python /mnt/sweep/sweep.py \
    --sweep_def /mnt/sweep/final_report/gru_seq_len.json

echo "Sweep agent completed"

# Clean up temporary files
# Only the last job in the array should perform cleanup
# This ensures we don't delete files while other jobs are still using them
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
  echo "This is the last array job. Performing cleanup..."
  
  # Wait a bit to make sure all other jobs have finished
  sleep 30
  
  # Clean up the sweep ID file
  if [ -f "sweep/tmp/latest_sweep_id.txt" ]; then
    echo "Removing sweep ID file..."
    rm sweep/tmp/latest_sweep_id.txt
  fi
  
  # Remove any other temporary files in the tmp directory
  echo "Cleaning up temp directory..."
  find sweep/tmp -type f -name "temp_*" -delete
  
  # Compress log files to save space
  echo "Compressing log files..."
  find sweep/logs -name "*.log" -exec gzip -f {} \;
  
  echo "Cleanup completed"
else
  echo "Not the last array job, skipping cleanup"
fi

echo "Job finished at $(date -u +"%Y-%m-%d %H:%M:%S")"