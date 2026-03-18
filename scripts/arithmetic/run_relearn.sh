#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=relearn_sweep_shir
#SBATCH --output=logs/relearn_%j.out
#SBATCH --error=logs/relearn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

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

# --- Execute Relearning Sweep ---
echo "Starting Relearning Sweep on Node: $SLURMD_NODENAME"
echo "Target Setup: gemma-2-0.3B_train_only_forget"

python scripts/arithmetic/run_relearn.py --setups gemma-2-0.3B_train_only_forget

echo "Relearning Sweep Finished at $(date)"