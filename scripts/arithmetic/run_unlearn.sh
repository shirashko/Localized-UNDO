#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=unlearn_sweep
#SBATCH --output=logs/unlearn_%j.out
#SBATCH --error=logs/unlearn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

# --- Execute Unlearning Sweep ---
echo "Starting Unlearning Sweep on Node: $SLURMD_NODENAME"

# Note: Your python script uses launch_in_parallel_one_per_gpu.
# If you only request 1 GPU in SLURM (above), it will run the sweep
# sequentially (one experiment after another) on that single GPU.
python scripts/arithmetic/run_unlearn.py