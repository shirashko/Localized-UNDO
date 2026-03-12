#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_gemma_arithmetic
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Variables ---
# Switch to run the different setups
# 1. gemma-2-0.1B_all_arithmetic+eng
# 2. gemma-2-0.1B_addition_subtraction+eng
SETUP_ID="gemma-2-0.1B_all_arithmetic+eng"

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Navigate to Project Directory ---
cd /home/morg/students/rashkovits/Localized-UNDO
mkdir -p logs

# --- Execute Training ---
echo "Starting Job on Node: $SLURMD_NODENAME"
echo "Running Setup: $SETUP_ID"

nvidia-smi --query-gpu=name --format=csv,noheader

accelerate launch scripts/arithmetic/run_pretrain.py --setup "$SETUP_ID"