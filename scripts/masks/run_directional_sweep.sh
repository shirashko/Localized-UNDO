#!/bin/bash

#SBATCH --job-name=svd_directional_sweep
#SBATCH --output=logs/mask_sweep_%j.out
#SBATCH --error=logs/mask_sweep_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # Reduced from 16 to get scheduled faster
#SBATCH --mem=80G                  # Reduced from 128G for better scheduling
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
# Use 'conda activate' or 'source activate' depending on your cluster setup
source activate undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- Debugging Paths (CRITICAL) ---
# This will print to your .out file exactly what the script sees
echo "Working Directory: $(pwd)"
echo "Checking if model directory exists:"
ls -ld /home/morg/students/rashkovits/Localized-UNDO/models/non-wmdp/pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model

# --- Execute ---
echo "Starting directional sweep on Node: $SLURMD_NODENAME"
# Use python -u for unbuffered output (so you see logs in real-time)
python -u scripts/masks/run_directional_sweep.py

echo "Sweep Finished at $(date)"