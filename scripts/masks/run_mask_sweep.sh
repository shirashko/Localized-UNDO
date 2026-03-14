#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=mask_sweep_diag
#SBATCH --output=logs/mask_sweep_%j.out
#SBATCH --error=logs/mask_sweep_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1              # 1 GPU is enough for inference-based diagnostics
#SBATCH --cpus-per-task=16         # High CPU count helps with mask math and data loading
#SBATCH --mem=128G                 # Loading two models + masks into RAM requires significant memory

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- Execute Mask Generation and Diagnostic Sweep ---
echo "Starting Mask Sweep and Diagnostics on Node: $SLURMD_NODENAME"
echo "Time: $(date)"

python scripts/masks/run_mask_sweep.py

echo "Sweep and Diagnostics Finished at $(date)"