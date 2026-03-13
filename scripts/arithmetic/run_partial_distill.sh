#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=distill_sweep
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
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

# --- Execute Partial Distillation Sweep ---
echo "Starting Partial Distillation Sweep on Node: $SLURMD_NODENAME"

# We pass the setup ID defined in your YAML 'setups' section
python scripts/arithmetic/run_partial_distill.py --setup gemma-2-0.1B_MaxEnt